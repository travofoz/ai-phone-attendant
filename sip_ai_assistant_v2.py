import pjsua2 as pj
import time
import threading
import wave
import io
import numpy as np
import logging
import queue
import configparser
import os
import sys
import re
import pandas as pd
import webrtcvad

# --- API Clients ---
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from gtts import gTTS
from pydub import AudioSegment

# --- Constants & Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger("SIP_AI_Assistant")
CONFIG_FILE = 'config.ini'

# Read configuration
if not os.path.exists(CONFIG_FILE):
    logger.error(f"Configuration file '{CONFIG_FILE}' not found.")
    sys.exit(1)

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

try:
    # SIP Config
    SIP_REGISTRAR = config['SIP']['registrar']
    SIP_USER = config['SIP']['username']
    SIP_PASSWORD = config['SIP']['password']
    SIP_REALM = config['SIP'].get('realm', '*')

    # API Keys
    OPENAI_API_KEY = config['API_KEYS']['openai_api_key']
    GEMINI_API_KEY = config['API_KEYS']['gemini_api_key']

    # Audio Config
    PJSIP_SAMPLE_RATE = int(config['AUDIO']['sample_rate'])
    if PJSIP_SAMPLE_RATE not in [8000, 16000, 32000, 48000]:
         logger.warning(f"PJSIP Sample Rate {PJSIP_SAMPLE_RATE} might not be optimal or supported by all components (Whisper prefers 16k, VAD supports 8k, 16k, 32k).")

    # VAD Config
    VAD_AGGRESSIVENESS = int(config['VAD']['aggressiveness'])
    VAD_FRAME_MS = int(config['VAD']['frame_ms'])
    VAD_SILENCE_PADDING_FRAMES = int(config['VAD']['silence_padding_ms'] / VAD_FRAME_MS)
    VAD_SPEECH_MIN_FRAMES = int(config['VAD']['speech_min_ms'] / VAD_FRAME_MS)
    VAD_BYTES_PER_FRAME = (PJSIP_SAMPLE_RATE // 1000) * VAD_FRAME_MS * 2 # 16-bit PCM = 2 bytes/sample

    # CSV Paths
    CSV_DIR_PATH = config['CSV_PATHS']['directory']
    CSV_COSTS_PATH = config['CSV_PATHS']['costs']
    CSV_LEAD_TIMES_PATH = config['CSV_PATHS']['lead_times']

    # Transfer Config
    DEFAULT_TRANSFER_TARGET = config['TRANSFER']['default_target_uri']

    # Prompting Config
    SYSTEM_PROMPT = config['PROMPTING']['system_prompt']

except KeyError as e:
    logger.error(f"Missing configuration key in {CONFIG_FILE}: {e}")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid numerical value in {CONFIG_FILE}: {e}")
    sys.exit(1)


# --- Initialize API Clients ---
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    # Configure safety settings for Gemini if needed (adjust as necessary)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    gemini_model = genai.GenerativeModel(
        'gemini-1.5-flash', # Use a model that supports function calling well
         safety_settings=safety_settings
        # system_instruction=SYSTEM_PROMPT # System prompt is now passed differently with history
        )
except Exception as e:
    logger.error(f"Failed to initialize API clients: {e}")
    sys.exit(1)

# --- DataManager for CSV Tools ---
class DataManager:
    def __init__(self, dir_path, costs_path, lead_times_path):
        self.dir_df = self._load_csv(dir_path)
        self.costs_df = self._load_csv(costs_path)
        self.lead_times_df = self._load_csv(lead_times_path)
        logger.info("DataManager initialized.")

    def _load_csv(self, path):
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded CSV: {path}")
            # Basic validation: Check for empty file
            if df.empty:
                 logger.warning(f"CSV file is empty: {path}")
            return df
        except FileNotFoundError:
            logger.error(f"CSV file not found: {path}")
            return pd.DataFrame() # Return empty DataFrame on error
        except pd.errors.EmptyDataError:
             logger.error(f"CSV file is empty or invalid: {path}")
             return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading CSV {path}: {e}")
            return pd.DataFrame()

    def lookup_employee(self, name: str) -> str:
        """Looks up an employee in the company directory by name."""
        if self.dir_df.empty:
            return "I cannot access the company directory data right now."
        try:
            # Case-insensitive search
            result = self.dir_df[self.dir_df['Name'].str.contains(name, case=False, na=False)]
            if not result.empty:
                info = result.iloc[0] # Take the first match
                # Construct SIP URI from extension if available and domain is known
                sip_uri = None
                if 'Extension' in info and pd.notna(info['Extension']):
                     # Attempt to build a plausible SIP URI - THIS IS AN ASSUMPTION
                     # You might need a more robust way to map extensions to URIs
                     domain = SIP_REGISTRAR.split('@')[-1] if '@' in SIP_REGISTRAR else 'your_domain.com' # Guess domain
                     try:
                         ext_num = int(info['Extension']) # Ensure it's numeric if possible
                         sip_uri = f"sip:{ext_num}@{domain}"
                     except ValueError:
                         sip_uri = f"sip:{info['Extension']}@{domain}" # Use as string if not purely numeric


                details = f"Found {info['Name']} in {info.get('Department', 'N/A')}. "
                if sip_uri:
                     details += f"Their contact extension seems to be {info.get('Extension', 'N/A')}. Would you like me to transfer you? Use this for transfer: [TRANSFER:{sip_uri}]"
                elif 'Email' in info and pd.notna(info['Email']):
                     details += f"Their email is {info['Email']}. I can't directly transfer but you could email them."
                else:
                    details += "No direct transfer information available."
                return details
            else:
                return f"Sorry, I couldn't find anyone named '{name}' in the directory."
        except Exception as e:
            logger.error(f"Error during employee lookup for '{name}': {e}")
            return "I encountered an error while searching the directory."

    def get_part_cost(self, part_number: str) -> str:
        """Gets the estimated cost for a specific part number."""
        if self.costs_df.empty:
            return "I cannot access the cost sheet data right now."
        try:
            result = self.costs_df[self.costs_df['PartNumber'].str.lower() == part_number.lower()]
            if not result.empty:
                info = result.iloc[0]
                cost = info.get('CostPerUnit', 'N/A')
                unit = info.get('Unit', '')
                # Ensure cost is formatted reasonably
                try:
                    cost_val = float(cost)
                    cost_str = f"${cost_val:.2f}"
                except (ValueError, TypeError):
                    cost_str = str(cost) # Use original string if not a number

                return f"The estimated cost for {info.get('Description', part_number)} ({part_number}) is {cost_str} per {unit}. Remember, this is a preliminary estimate for the part only and doesn't include setup, labor, or shipping unless specified. A formal quote is needed for exact pricing."
            else:
                return f"Sorry, I couldn't find a cost estimate for part number '{part_number}'. It might be a custom item or not on my list. Please provide specifications for a formal quote."
        except Exception as e:
            logger.error(f"Error during cost lookup for '{part_number}': {e}")
            return "I encountered an error while searching the cost sheet."

    def calculate_lead_time(self, item_type: str, operations: list[str] = None, is_rush: bool = False, complexity_details: str = None) -> str:
        """Calculates an estimated lead time based on item type, operations, and complexity."""
        if self.lead_times_df.empty:
             return "I cannot access the lead time data right now."
        try:
            total_base_days = 0
            max_complexity_factor = 1.0
            notes = []

            # Base item type lookup
            item_result = self.lead_times_df[self.lead_times_df['ItemType'].str.contains(item_type, case=False, na=False)]
            if not item_result.empty:
                 base_item = item_result.iloc[0]
                 base_days = base_item.get('BaseDays', 0)
                 if pd.notna(base_days): total_base_days += float(base_days)
                 comp_factor = base_item.get('ComplexityFactor', 1.0)
                 if pd.notna(comp_factor): max_complexity_factor = max(max_complexity_factor, float(comp_factor))
                 if pd.notna(base_item.get('Notes')): notes.append(base_item['Notes'])

            # Add time for specific operations if provided
            if operations:
                for op in operations:
                     op_result = self.lead_times_df[
                         (self.lead_times_df['Operation'].str.contains(op, case=False, na=False)) &
                         # Optional: Filter by ItemType too if operations are specific
                         (self.lead_times_df['ItemType'].str.contains(item_type, case=False, na=False) | self.lead_times_df['ItemType'].isna()) # Allow general operations
                         ]
                     if not op_result.empty:
                          op_info = op_result.iloc[0]
                          op_days = op_info.get('BaseDays', 0)
                          if pd.notna(op_days): total_base_days += float(op_days)
                          op_comp_factor = op_info.get('ComplexityFactor', 1.0)
                          if pd.notna(op_comp_factor): max_complexity_factor = max(max_complexity_factor, float(op_comp_factor))
                          if pd.notna(op_info.get('Notes')): notes.append(op_info['Notes'])

            if total_base_days == 0 and not notes:
                 return f"I couldn't find specific lead time information for '{item_type}' with operations '{operations}'. This might require a custom quote based on your exact needs."


            # Apply complexity (using max factor found)
            estimated_days = total_base_days * max_complexity_factor

            # Apply rush factor if requested
            rush_factor = 1.0
            if is_rush:
                 rush_result = self.lead_times_df[self.lead_times_df['ItemType'].str.contains("Rush Order", case=False, na=False)]
                 if not rush_result.empty:
                     rush_factor = rush_result.iloc[0].get('Expedite Fee Factor', 1.0) # This column name is an assumption
                     if pd.isna(rush_factor) or float(rush_factor) <= 0: rush_factor = 1.5 # Default rush if not found or invalid
                     else: rush_factor = float(rush_factor)
                     estimated_days /= rush_factor # Assuming rush factor REDUCES time
                     notes.append("Rush order applied (estimated time reduced). Expedite fees will apply.")
                 else:
                     notes.append("Rush order requested, but couldn't find specific factor; standard estimates used.")

            # Add standard shipping estimate if not included
            if 'shipping' not in (op.lower() for op in operations or []):
                 ship_result = self.lead_times_df[self.lead_times_df['Operation'].str.contains("Shipping", case=False, na=False)]
                 if not ship_result.empty:
                      ship_days = ship_result.iloc[0].get('BaseDays', 0)
                      if pd.notna(ship_days): estimated_days += float(ship_days)


            response = f"Based on '{item_type}' "
            if operations: response += f"with operations '{', '.join(operations)}', "
            response += f"the estimated lead time is roughly {estimated_days:.1f} business days. "
            if complexity_details: response += f"Complexity note: {complexity_details}. "
            if notes: response += f"Notes: {'; '.join(filter(None, notes))}. "
            response += "This is a preliminary estimate. Actual lead time depends on current workload, material availability, and final specifications provided in a formal quote request."

            return response

        except Exception as e:
            logger.error(f"Error during lead time calculation for '{item_type}': {e}")
            return "I encountered an error while calculating the lead time."


# --- Gemini Tools Definition ---
data_manager = DataManager(CSV_DIR_PATH, CSV_COSTS_PATH, CSV_LEAD_TIMES_PATH)

tools = [
    {
        "function_declarations": [
            {
                "name": "lookup_employee",
                "description": "Looks up an employee in the company directory by name to find their department or contact info.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING", "description": "The name (or partial name) of the employee to look up."}
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "get_part_cost",
                "description": "Gets the estimated base cost for a standard part number from the cost sheet.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "part_number": {"type": "STRING", "description": "The specific part number (e.g., PN-1001)."}
                    },
                    "required": ["part_number"]
                }
            },
            {
                "name": "calculate_lead_time",
                "description": "Calculates an estimated lead time based on item type, operations involved, and optionally complexity or rush status.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "item_type": {"type": "STRING", "description": "The general type of item (e.g., 'Widget', 'Custom Part', 'Standard Order')."},
                        "operations": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "A list of manufacturing or processing operations involved (e.g., ['Machining', 'Assembly', 'Shipping']). Optional."
                        },
                         "is_rush": {"type": "BOOLEAN", "description": "Set to true if the caller requests a rush order. Optional, defaults to false."},
                         "complexity_details": {"type": "STRING", "description": "Any specific details mentioned by the caller regarding complexity. Optional."}
                    },
                    "required": ["item_type"]
                }
            }
        ]
    }
]

# Map function names to actual Python functions
available_functions = {
    "lookup_employee": data_manager.lookup_employee,
    "get_part_cost": data_manager.get_part_cost,
    "calculate_lead_time": data_manager.calculate_lead_time,
}


# --- Global Queues and Events ---
playback_queues = {} # Dictionary: {call_id: Queue()}
processing_signals = {} # Dictionary: {call_id: Event()} # Signal to start processing VAD buffer
transfer_requests = {} # Dictionary: {call_id: Queue()} # Queue for transfer URIs


# --- Audio Processing Thread ---
def audio_processing_worker(call_id, initial_history):
    """Thread function to handle VAD, STT, LLM (with tools), and TTS for a specific call."""
    logger.info(f"[Call {call_id}] Audio processing worker started.")
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    speech_buffer = bytearray()
    silence_frames = 0
    speech_frames = 0
    is_speaking = False
    history = list(initial_history) # Copy initial history for this call

    # Make sure queues/events exist for this call
    if call_id not in playback_queues: playback_queues[call_id] = queue.Queue()
    if call_id not in processing_signals: processing_signals[call_id] = queue.Queue() # Use queue to pass audio data
    if call_id not in transfer_requests: transfer_requests[call_id] = queue.Queue()

    while True:
        try:
            # Wait for audio data chunk from VAD processor in MyAudioMediaPort
            raw_pcm_chunk = processing_signals[call_id].get(timeout=60) # Timeout prevents hanging forever
            if raw_pcm_chunk is None: # Signal to stop
                logger.info(f"[Call {call_id}] Audio processing worker received stop signal.")
                break
            if not isinstance(raw_pcm_chunk, bytes) or len(raw_pcm_chunk) != VAD_BYTES_PER_FRAME:
                 logger.warning(f"[Call {call_id}] Received invalid audio chunk. Size: {len(raw_pcm_chunk)}, Expected: {VAD_BYTES_PER_FRAME}")
                 continue

            # --- VAD Logic ---
            try:
                is_speech = vad.is_speech(raw_pcm_chunk, PJSIP_SAMPLE_RATE)
            except Exception as vad_error:
                 logger.error(f"[Call {call_id}] VAD error processing frame: {vad_error}")
                 continue # Skip this frame

            if is_speech:
                # logger.debug(f"[Call {call_id}] Speech detected.")
                speech_buffer.extend(raw_pcm_chunk)
                speech_frames += 1
                silence_frames = 0
                if speech_frames >= VAD_SPEECH_MIN_FRAMES:
                    is_speaking = True # Mark as speaking only after minimum duration
            else:
                # logger.debug(f"[Call {call_id}] Silence detected.")
                if is_speaking: # Only buffer silence if we were recently speaking
                     speech_buffer.extend(raw_pcm_chunk) # Add padding silence
                     silence_frames += 1
                     if silence_frames >= VAD_SILENCE_PADDING_FRAMES:
                         logger.info(f"[Call {call_id}] End of speech detected after {speech_frames} speech frames and {silence_frames} silence frames.")
                         # --- Process the buffered speech ---
                         if len(speech_buffer) > VAD_BYTES_PER_FRAME * VAD_SPEECH_MIN_FRAMES : # Ensure buffer isn't just noise/padding
                             process_audio_buffer(call_id, bytes(speech_buffer), history) # Pass current history
                         # Reset after processing
                         speech_buffer = bytearray()
                         silence_frames = 0
                         speech_frames = 0
                         is_speaking = False
                else:
                     # Silence before minimum speech duration, discard buffer
                     speech_buffer = bytearray()
                     silence_frames = 0
                     speech_frames = 0


        except queue.Empty:
            # Timeout occurred, check if call is still active or just idle
            # logger.debug(f"[Call {call_id}] Processing queue timeout.")
            # If there's lingering speech in buffer without enough silence, process it?
            if is_speaking and len(speech_buffer) > VAD_BYTES_PER_FRAME * VAD_SPEECH_MIN_FRAMES :
                 logger.info(f"[Call {call_id}] Processing lingering speech buffer due to inactivity.")
                 process_audio_buffer(call_id, bytes(speech_buffer), history)
                 speech_buffer = bytearray()
                 silence_frames = 0
                 speech_frames = 0
                 is_speaking = False
            continue # Continue waiting
        except Exception as e:
            logger.error(f"[Call {call_id}] Error in audio processing worker loop: {e}")
            # Reset state potentially
            speech_buffer = bytearray()
            silence_frames = 0
            speech_frames = 0
            is_speaking = False
            time.sleep(0.5) # Avoid busy-looping on errors

    logger.info(f"[Call {call_id}] Audio processing worker stopped.")
    # Clean up queues for this call ID
    if call_id in playback_queues: del playback_queues[call_id]
    if call_id in processing_signals: del processing_signals[call_id]
    if call_id in transfer_requests: del transfer_requests[call_id]


def process_audio_buffer(call_id, audio_data, history):
    """Handles STT, LLM, and TTS for a given audio buffer."""
    logger.info(f"[Call {call_id}] Processing {len(audio_data)} bytes of audio...")
    if not audio_data:
        return

    # 1. Prepare WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(PJSIP_SAMPLE_RATE)
        wf.writeframes(audio_data)
    wav_buffer.seek(0)

    # 2. STT (Whisper)
    try:
        logger.info(f"[Call {call_id}] Sending audio to Whisper...")
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=("input.wav", wav_buffer, "audio/wav")
        )
        caller_text = transcript.text
        logger.info(f"[Call {call_id}] Whisper Transcription: {caller_text}")

        if not caller_text.strip():
            logger.info(f"[Call {call_id}] Empty transcription, skipping Gemini.")
            return

        # Add user message to history
        history.append({'role': 'user', 'parts': [{'text': caller_text}]})
        # Limit history length (e.g., last 10 turns = 20 messages)
        if len(history) > 20:
             history = [history[0]] + history[-19:] # Keep system prompt + last messages


    except Exception as e:
        logger.error(f"[Call {call_id}] Whisper API error: {e}")
        # Maybe play an error message?
        play_error_message(call_id, "Sorry, I had trouble understanding that.")
        # Remove the failed user message from history
        if history and history[-1]['role'] == 'user':
            history.pop()
        return

    # 3. LLM (Gemini with Function Calling)
    try:
        logger.info(f"[Call {call_id}] Sending text to Gemini with history and tools...")

        # === Gemini API Call ===
        gemini_response = gemini_model.generate_content(
            history, # Pass the whole history
            tools=tools,
            generation_config=genai.types.GenerationConfig(temperature=0.7) # Adjust temperature as needed
        )

        # === Handle Gemini Response (Function Call or Text) ===
        response_message = gemini_response.candidates[0].content

        if response_message.parts[0].function_call.name:
            # === Function Call Requested ===
            fc = response_message.parts[0].function_call
            function_name = fc.name
            args = fc.args

            logger.info(f"[Call {call_id}] Gemini requested function call: {function_name} with args: {dict(args)}")

            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                try:
                    # Call the actual Python function
                    function_response_content = function_to_call(**dict(args))
                    logger.info(f"[Call {call_id}] Function {function_name} executed. Result: {function_response_content}")

                    # === Send Function Result Back to Gemini ===
                    history.append(response_message) # Add Gemini's function call request to history
                    history.append({
                        "role": "function", # Special role for function results
                        "parts": [
                            {"function_response": {
                                "name": function_name,
                                "response": {"content": function_response_content}
                                }
                            }
                        ]
                    })

                    # Make second call to Gemini with the function result
                    logger.info(f"[Call {call_id}] Sending function response back to Gemini...")
                    second_response = gemini_model.generate_content(history) # Pass updated history

                    # Check if the second response is valid text
                    if not second_response.candidates or not second_response.candidates[0].content.parts:
                         raise Exception("Gemini returned empty response after function call.")

                    final_text_response = second_response.candidates[0].content.parts[0].text
                    logger.info(f"[Call {call_id}] Gemini Final Response (after function call): {final_text_response}")
                     # Add Gemini's final response to history
                    history.append(second_response.candidates[0].content)


                except Exception as func_exec_e:
                    logger.error(f"[Call {call_id}] Error executing function {function_name} or calling Gemini again: {func_exec_e}")
                    final_text_response = f"Sorry, I encountered an error trying to use my tool: {function_name}. Please try asking differently."
                    # Add error message to history as model response
                    history.append({'role': 'model', 'parts': [{'text': final_text_response}]})

            else:
                logger.warning(f"[Call {call_id}] Gemini requested unknown function: {function_name}")
                final_text_response = f"Sorry, I don't know how to perform the action: {function_name}."
                # Add error message to history as model response
                history.append({'role': 'model', 'parts': [{'text': final_text_response}]})

        else:
            # === Direct Text Response ===
            final_text_response = response_message.parts[0].text
            logger.info(f"[Call {call_id}] Gemini Direct Response: {final_text_response}")
             # Add Gemini's direct response to history
            history.append(response_message)


        # 4. Check for Transfer Request in Response
        transfer_match = re.search(r"\[TRANSFER:(sip:[^\]]+)\]", final_text_response)
        if transfer_match:
            target_uri = transfer_match.group(1)
            logger.info(f"[Call {call_id}] Transfer requested in response to: {target_uri}")
            # Remove the tag from the spoken response
            final_text_response = re.sub(r"\[TRANSFER:[^\]]+\]", "", final_text_response).strip()
            # Add a preamble before transferring
            final_text_response = f"Okay, I can transfer you now. Please wait a moment. {final_text_response}" # Add preamble
            # Queue the transfer request *after* saying the message
            # We'll handle this after TTS playback finishes in this cycle
            transfer_uri_to_request = target_uri # Store it
        else:
            transfer_uri_to_request = None

        # 5. TTS (gTTS) & Format Conversion
        if final_text_response: # Ensure there is text to speak
             generate_and_queue_tts(call_id, final_text_response)

             # If a transfer was requested, queue it now AFTER queuing the TTS.
             # The actual transfer will happen in the main thread check.
             if transfer_uri_to_request:
                  if call_id in transfer_requests:
                      transfer_requests[call_id].put(transfer_uri_to_request)
                  else:
                       logger.warning(f"[Call {call_id}] Transfer request queue not found, cannot initiate transfer.")

    except Exception as e:
        logger.error(f"[Call {call_id}] Gemini API or response processing error: {e}")
        play_error_message(call_id, "Sorry, I had trouble generating a response.")
        # Remove the potentially problematic Gemini response from history if added
        if history and history[-1]['role'] == 'model':
             history.pop()


def generate_and_queue_tts(call_id, text):
    """Generates TTS audio using gTTS and queues it for playback."""
    try:
        if not text.strip():
            return
        logger.info(f"[Call {call_id}] Generating TTS for: \"{text[:50]}...\"")
        tts_mp3_buffer = io.BytesIO()
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(tts_mp3_buffer)
        tts_mp3_buffer.seek(0)

        audio_segment = AudioSegment.from_file(tts_mp3_buffer, format="mp3")
        audio_segment = audio_segment.set_frame_rate(PJSIP_SAMPLE_RATE)
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_sample_width(2) # 16-bit

        tts_pcm_bytes = audio_segment.raw_data

        # Queue for playback (ensure queue exists)
        if call_id in playback_queues:
            # Chunk the TTS data into smaller frames for smoother playback
            playback_frame_size = (PJSIP_SAMPLE_RATE // 50) * 2 # 20ms frames
            for i in range(0, len(tts_pcm_bytes), playback_frame_size):
                 chunk = tts_pcm_bytes[i:i + playback_frame_size]
                 playback_queues[call_id].put(chunk)
            logger.info(f"[Call {call_id}] Queued {len(tts_pcm_bytes)} bytes of TTS audio for playback.")
        else:
             logger.warning(f"[Call {call_id}] Playback queue not found, cannot play TTS.")

    except Exception as e:
        logger.error(f"[Call {call_id}] TTS generation or queuing error: {e}")

def play_error_message(call_id, message="Sorry, an error occurred."):
     """Plays a standard error message."""
     logger.info(f"[Call {call_id}] Playing error message: {message}")
     generate_and_queue_tts(call_id, message)


# --- Custom Audio Media Port ---
class MyAudioMediaPort(pj.AudioMediaPort):
    def __init__(self, call_id):
        pj.AudioMediaPort.__init__(self)
        self.call_id = call_id
        self.vad_frame_buffer = bytearray()
        self.playback_buffer = b'' # Holds current chunk from playback_queue
        logger.info(f"[Call {call_id}] MyAudioMediaPort created")

    def onFrameReceived(self, frame):
        """Called by PJSIP with incoming audio. Sends fixed-size chunks to VAD thread."""
        try:
            # Ensure frame data is bytes
            frame_data = bytes(frame.data)
            self.vad_frame_buffer.extend(frame_data)

            # Process in VAD_FRAME_MS chunks
            while len(self.vad_frame_buffer) >= VAD_BYTES_PER_FRAME:
                chunk = self.vad_frame_buffer[:VAD_BYTES_PER_FRAME]
                del self.vad_frame_buffer[:VAD_BYTES_PER_FRAME]

                # Send chunk to the processing thread via its queue
                if self.call_id in processing_signals:
                    try:
                        processing_signals[self.call_id].put_nowait(bytes(chunk))
                    except queue.Full:
                        logger.warning(f"[Call {self.call_id}] VAD processing queue is full, dropping frame.")
                    except Exception as qe:
                         logger.error(f"[Call {self.call_id}] Error putting frame to VAD queue: {qe}")

                else:
                    logger.warning(f"[Call {self.call_id}] Processing signal queue not found.")

        except Exception as e:
            logger.error(f"[Call {self.call_id}] Error in onFrameReceived: {e}")

        return True # Indicate frame was processed

    def onFrameRequested(self, frame):
        """Called by PJSIP when it needs audio TO SEND."""
        if self.call_id not in playback_queues:
             # logger.warning(f"[Call {self.call_id}] Playback queue missing in onFrameRequested.")
             frame.data[:] = b'\x00' * frame.size # Send silence
             return True

        target_size = frame.size

        # Fill frame data from playback_buffer first
        if len(self.playback_buffer) >= target_size:
             frame.data[:] = self.playback_buffer[:target_size]
             self.playback_buffer = self.playback_buffer[target_size:]
             return True
        else:
             # Need more data, try getting from queue
             frame.data[:len(self.playback_buffer)] = self.playback_buffer
             filled_size = len(self.playback_buffer)
             self.playback_buffer = b'' # Consumed partial buffer

             while filled_size < target_size:
                  try:
                      new_chunk = playback_queues[self.call_id].get_nowait()
                      remaining_needed = target_size - filled_size
                      if len(new_chunk) <= remaining_needed:
                          frame.data[filled_size : filled_size + len(new_chunk)] = new_chunk
                          filled_size += len(new_chunk)
                      else: # new_chunk is larger than needed
                          frame.data[filled_size : target_size] = new_chunk[:remaining_needed]
                          self.playback_buffer = new_chunk[remaining_needed:] # Save leftover
                          filled_size = target_size # Frame is full
                          break # Exit while loop

                  except queue.Empty:
                      # No more TTS data, fill remaining with silence
                      # logger.debug(f"[Call {self.call_id}] Playback queue empty, padding with silence.")
                      frame.data[filled_size:target_size] = b'\x00' * (target_size - filled_size)
                      filled_size = target_size # Frame is full
                      break # Exit while loop
                  except Exception as e:
                      logger.error(f"[Call {self.call_id}] Error getting from playback queue: {e}")
                      frame.data[filled_size:target_size] = b'\x00' * (target_size - filled_size)
                      filled_size = target_size # Frame is full
                      break # Exit while loop

        return True


# --- PJSIP Call Class ---
class MyCall(pj.Call):
    def __init__(self, acc, call_id=pj.PJSUA_INVALID_ID):
        pj.Call.__init__(self, acc, call_id)
        self.acc = acc
        self.my_audio_media_port = None
        self.processing_thread = None
        # Initialize conversation history with the system prompt
        self.conversation_history = [{'role': 'user', 'parts': [{'text': 'placeholder'}]}, # Placeholder for first user turn
                                     {'role': 'model', 'parts': [{'text': SYSTEM_PROMPT}]}] # System prompt
        logger.info(f"[Call {self.id}] Incoming call for account {acc.info().uri}")

    def onCallState(self, prm):
        ci = self.getInfo()
        logger.info(f"[Call {self.id}] State: {ci.stateText} (Last status: {ci.lastCode} {ci.lastReason})")

        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            logger.info(f"[Call {self.id}] Disconnected.")
            self.cleanup_call()

        elif ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            logger.info(f"[Call {self.id}] Active.")
            # Media setup happens in onCallMediaState

    def onCallMediaState(self, prm):
        ci = self.getInfo()
        logger.info(f"[Call {self.id}] Media state changed.")
        for i, med in enumerate(ci.media):
            logger.info(f"  Media {i}: type={med.type}, status={med.status}")
            if med.type == pj.PJMEDIA_TYPE_AUDIO and med.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                logger.info(f"Audio media for call {self.id} is now ACTIVE.")
                if not self.my_audio_media_port: # Ensure not already set up
                    try:
                        aud_med = self.getAudioMedia(i)
                        self.my_audio_media_port = MyAudioMediaPort(self.id)
                        ep = pj.Endpoint.instance()
                        ep.audDevManager().getCaptureDevMedia().startTransmit(self.my_audio_media_port)
                        self.my_audio_media_port.startTransmit(aud_med)
                        logger.info(f"[Call {self.id}] Connected MyAudioMediaPort.")

                        # Start the processing thread for this call
                        if not self.processing_thread or not self.processing_thread.is_alive():
                             # Pass a *copy* of the initial history structure
                             initial_hist = [
                                 {'role': 'system', 'parts': [{'text': SYSTEM_PROMPT}]}
                             ]
                             self.processing_thread = threading.Thread(target=audio_processing_worker,
                                                                         args=(self.id, initial_hist), # Pass call ID and history
                                                                         daemon=True)
                             self.processing_thread.start()
                        else:
                             logger.warning(f"[Call {self.id}] Processing thread already exists.")


                    except Exception as e:
                        logger.error(f"[Call {self.id}] Error setting up media: {e}")
                        self.hangup_call(500, "Media Error")
            elif med.type == pj.PJMEDIA_TYPE_AUDIO and med.status == pj.PJSUA_CALL_MEDIA_NONE:
                 logger.info(f"Audio media for call {self.id} is now NONE.")
                 # Media stopped, potentially clean up port? PJSIP might handle this.
                 pass

    def request_transfer(self, target_uri):
        """Initiates a call transfer (REFER)."""
        if not self.isActive():
            logger.warning(f"[Call {self.id}] Cannot transfer inactive call.")
            return False
        try:
            logger.info(f"[Call {self.id}] Attempting to transfer call to {target_uri}")
            xfer_param = pj.CallOpParam()
            xfer_param.statusCode = 202 # Accepted
            self.xfer(target_uri, xfer_param)
            # Note: The call state might change shortly after this.
            # The 'onCallState' will handle the DISCONNECTED state eventually.
            logger.info(f"[Call {self.id}] REFER message sent to {target_uri}.")
            return True
        except pj.Error as e:
            logger.error(f"[Call {self.id}] Failed to initiate transfer to {target_uri}: {e.info()}")
            # Maybe play an error message to the user still on the line?
            play_error_message(self.id, "Sorry, I couldn't transfer the call.")
            return False
        except Exception as e:
             logger.error(f"[Call {self.id}] Unexpected error during transfer: {e}")
             play_error_message(self.id, "Sorry, an unexpected error occurred during transfer.")
             return False


    def hangup_call(self, code=200, reason="OK"):
        """Hangs up the call with a specific code and reason."""
        if self.isActive():
            logger.info(f"[Call {self.id}] Hanging up call (Code: {code} Reason: {reason})")
            hangup_prm = pj.CallOpParam()
            hangup_prm.statusCode = code
            hangup_prm.reason = reason
            try:
                 self.hangup(hangup_prm)
            except pj.Error as e:
                 logger.error(f"[Call {self.id}] Error during hangup: {e.info()}")
        else:
            logger.info(f"[Call {self.id}] Call already inactive, no hangup needed.")
        self.cleanup_call() # Ensure cleanup happens even if hangup fails


    def cleanup_call(self):
        """Cleans up resources associated with this call."""
        logger.info(f"[Call {self.id}] Cleaning up call resources.")
        # Stop the processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info(f"[Call {self.id}] Signaling processing thread to stop.")
            if self.id in processing_signals:
                processing_signals[self.id].put(None) # Send stop signal
                self.processing_thread.join(timeout=2) # Wait briefly
                if self.processing_thread.is_alive():
                    logger.warning(f"[Call {self.id}] Processing thread did not stop gracefully.")
            self.processing_thread = None

        # Clean up media port (be careful with PJSIP object lifecycle)
        if self.my_audio_media_port:
             logger.info(f"[Call {self.id}] Cleaning up audio media port.")
             try:
                 # Try to stop transmissions cleanly
                 ep = pj.Endpoint.instance()
                 # Check if audio media still exists for the call
                 # This check avoids errors if media was already torn down
                 aud_med = None
                 ci = self.getInfo() # Get latest info
                 for i, med in enumerate(ci.media):
                      if med.type == pj.PJMEDIA_TYPE_AUDIO:
                           try:
                                aud_med = self.getAudioMedia(i)
                                break
                           except pj.Error: # May fail if call/media is gone
                                pass

                 if aud_med:
                     ep.audDevManager().getCaptureDevMedia().stopTransmit(self.my_audio_media_port)
                     self.my_audio_media_port.stopTransmit(aud_med)
                 # De-register the port from the sound device manager
                 # ep.audDevManager().removeAudioMedia(self.my_audio_media_port) # Causes issues sometimes
             except pj.Error as e:
                  logger.warning(f"[Call {self.id}] PJSIP error stopping media transmission during cleanup: {e.info()}")
             except AttributeError:
                 logger.warning(f"[Call {self.id}] Could not get Endpoint instance during cleanup, already shut down?")
             except Exception as e:
                 logger.warning(f"[Call {self.id}] Error cleaning up media port: {e}")
             finally:
                 # Explicitly break circular references if necessary, though Python's GC + PJSIP's handling *should* work
                 # Be cautious deleting pjsua2 objects directly, it can lead to segfaults.
                 # Relying on PJSIP to clean up might be safer.
                 self.my_audio_media_port = None

        # Clear any remaining items in global queues for this call ID
        if self.id in playback_queues: del playback_queues[self.id]
        if self.id in processing_signals: del processing_signals[self.id]
        if self.id in transfer_requests: del transfer_requests[self.id]

        logger.info(f"[Call {self.id}] Cleanup complete.")
        # Inform the account that this call object can be released
        if self.acc:
            self.acc.notify_call_disconnected(self.id)


# --- PJSIP Account Class ---
class MyAccount(pj.Account):
    def __init__(self):
        pj.Account.__init__(self)
        self.active_calls = {} # Dictionary: {call_id: MyCall}

    def onRegState(self, prm):
        info = self.getInfo()
        logger.info(f"Account {info.uri} Reg State: {prm.reason} (Code: {prm.code}, Expires: {prm.expiration})")
        if prm.code != 200:
            logger.warning("SIP Registration failed. Check credentials/registrar. Retrying automatically?")

    def onIncomingCall(self, prm):
        call_id = prm.callId
        remote_uri = prm.remoteUri
        logger.info(f"Incoming call {call_id} from {remote_uri}")

        if call_id in self.active_calls:
             logger.warning(f"Call ID {call_id} already exists? Ignoring.")
             # Maybe reject the new INVITE?
             prm.code = 486 # Busy Here
             return

        call = MyCall(self, call_id)
        call_prm = pj.CallOpParam()

        try:
            # Auto-answer with 180 Ringing first, then 200 OK
            call_prm.statusCode = 180 # Ringing
            call.answer(call_prm)
            logger.info(f"[Call {call_id}] Sent 180 Ringing.")
            time.sleep(0.1) # Small delay sometimes helps

            call_prm.statusCode = 200 # OK
            call.answer(call_prm)
            self.active_calls[call_id] = call
            logger.info(f"[Call {call_id}] Answered incoming call from {remote_uri}")

        except pj.Error as e:
            logger.error(f"Failed to answer call {call_id}: {e.info()}")
            # No need to hangup explicitly if answer failed, PJSIP handles it.
            if call_id in self.active_calls: del self.active_calls[call_id]
        except Exception as e:
             logger.error(f"Unexpected error answering call {call_id}: {e}")
             if call_id in self.active_calls: del self.active_calls[call_id]

    def notify_call_disconnected(self, call_id):
        """Called by MyCall when it's cleaned up."""
        if call_id in self.active_calls:
            logger.info(f"Removing call {call_id} from active calls list.")
            del self.active_calls[call_id]
        else:
             logger.warning(f"Tried to remove non-existent call {call_id} from active calls.")

    def cleanup_calls(self):
        """Hang up and clean up all active calls for this account."""
        logger.info(f"Cleaning up active calls for account {self.getInfo().uri}...")
        # Iterate over a copy of the keys, as hangup_call modifies the dictionary
        call_ids = list(self.active_calls.keys())
        for call_id in call_ids:
             if call_id in self.active_calls:
                 call = self.active_calls[call_id]
                 call.hangup_call(603, "Service Shutdown") # 603 Decline
        self.active_calls.clear()
        logger.info("Finished cleaning up active calls.")


# --- Main Application Logic ---
class SipClientApp:
    def __init__(self):
        self.ep = None
        self.acc = None
        self.transport = None
        self.running = False
        self.check_transfer_thread = None
        self.stop_event = threading.Event()

    def start(self):
        logger.info("Starting SIP Client...")
        try:
            # 1. Create and Init Endpoint
            self.ep = pj.Endpoint()
            self.ep.libCreate()
            ep_cfg = pj.EpConfig()
            ep_cfg.uaConfig.userAgent = "Python Gemini Assistant v2.0 (Manufacturing)"
            ep_cfg.medConfig.clockRate = PJSIP_SAMPLE_RATE
            # Crucial for custom audio devices/ports: No automatic sound device connection
            ep_cfg.medConfig.noAutoTransmit = True
            ep_cfg.medConfig.noAutoPlay = True
            # Set higher logging level for PJSIP debugging if needed
            # ep_cfg.logConfig.level = 5
            # ep_cfg.logConfig.consoleLevel = 5
            self.ep.libInit(ep_cfg)

            # 2. Create Transport
            tp_cfg = pj.TransportConfig()
            # tp_cfg.port = 5060 # Or 0 for random
            self.transport = self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tp_cfg)
            logger.info(f"SIP Transport created on {self.transport.getInfo().localName}")

            # 3. Start PJSIP Library
            self.ep.libStart()
            logger.info("PJSIP Library Started.")

            # Set null sound device - we are handling audio via custom port
            self.ep.audDevManager().setNullDev()

            # 4. Configure and Create Account
            acc_cfg = pj.AccountConfig()
            # Construct account ID URI correctly
            reg_uri_parts = SIP_REGISTRAR.split(':')
            sip_domain = reg_uri_parts[-1] if len(reg_uri_parts) > 1 else SIP_REGISTRAR
            acc_cfg.idUri = f"sip:{SIP_USER}@{sip_domain}"
            acc_cfg.regConfig.registrarUri = SIP_REGISTRAR
            acc_cfg.sipConfig.authCreds.append(pj.AuthCredInfo(SIP_REALM, SIP_USER, SIP_PASSWORD))
            # Optional: Set proxy if needed (often same as registrar)
            # acc_cfg.sipConfig.proxies = [SIP_REGISTRAR]

            self.acc = MyAccount()
            self.acc.create(acc_cfg)
            logger.info("SIP Account created and registration initiated.")

            # 5. Start thread to check for transfer requests
            self.stop_event.clear()
            self.check_transfer_thread = threading.Thread(target=self.transfer_checker_worker, daemon=True)
            self.check_transfer_thread.start()


            self.running = True
            logger.info("SIP Client is running. Waiting for calls...")
            # Keep the main thread alive (PJSIP events are handled in background threads)
            while self.running and not self.stop_event.is_set():
                try:
                    # PJSIP handles events in its own threads. We just need to keep alive
                    # and potentially handle application-level logic like transfers.
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    logger.info("Ctrl+C received, initiating shutdown...")
                    self.stop()
                    break

        except pj.Error as e:
            logger.error(f"PJSIP Error during startup: {e.info()}")
            self.stop()
        except Exception as e:
            logger.error(f"General Error during startup: {e}", exc_info=True)
            self.stop()
        finally:
             if self.running: # Ensure stop is called if loop exits unexpectedly
                 self.stop()

    def transfer_checker_worker(self):
         """Periodically checks queues for transfer requests and initiates them."""
         logger.info("Transfer checker worker started.")
         while not self.stop_event.is_set():
             call_to_transfer = None
             transfer_uri = None
             call_id_to_process = None

             # Check all active calls for transfer requests
             if self.acc:
                  # Iterate safely over items
                  active_call_items = list(self.acc.active_calls.items())
                  for cid, call in active_call_items:
                       if cid in transfer_requests:
                            try:
                                 uri = transfer_requests[cid].get_nowait()
                                 logger.info(f"[Call {cid}] Found transfer request in queue for URI: {uri}")
                                 call_to_transfer = call
                                 transfer_uri = uri
                                 call_id_to_process = cid
                                 break # Process one transfer at a time
                            except queue.Empty:
                                 continue # No request for this call
                            except Exception as e:
                                 logger.error(f"[Call {cid}] Error checking transfer queue: {e}")

             if call_to_transfer and transfer_uri:
                  logger.info(f"[Call {call_id_to_process}] Initiating transfer via main thread...")
                  # Perform the actual PJSIP transfer operation
                  call_to_transfer.request_transfer(transfer_uri)
                  # The call state will change, and cleanup will happen via callbacks

             time.sleep(0.2) # Check periodically

         logger.info("Transfer checker worker stopped.")


    def stop(self):
        if not self.running:
            return
        logger.info("Shutting down SIP Client...")
        self.running = False
        self.stop_event.set() # Signal threads to stop

        # Stop the transfer checker thread
        if self.check_transfer_thread and self.check_transfer_thread.is_alive():
             logger.info("Stopping transfer checker thread...")
             self.check_transfer_thread.join(timeout=2)
             if self.check_transfer_thread.is_alive():
                  logger.warning("Transfer checker thread did not stop gracefully.")

        # PJSIP Cleanup
        try:
            if self.ep:
                logger.info("Starting PJSIP resource cleanup...")

                # 1. Hang up active calls first
                if self.acc:
                    self.acc.cleanup_calls() # This handles individual call cleanup now

                # 2. Destroy Account (unregister) - Needs to happen before transport/lib
                if self.acc:
                     logger.info("Deleting account...")
                     # PJSIP handles unregistration when account is deleted.
                     # Deleting the Python object `self.acc` triggers PJSIP's destruction.
                     # Explicitly calling `del self.acc` might be risky if references exist.
                     # Setting to None helps Python's GC.
                     # Let PJSIP's internal cleanup handle the account object after libDestroy.
                     # self.acc.delete() # Check pjsua2 docs if an explicit delete method exists/is needed
                     self.acc = None


                # 3. Destroy Transport (Needs lib running but no calls/accounts using it)
                 # Transport closing should ideally happen before libDestroy.
                 # Ensure lib is still running when closing transport.
                if self.transport:
                     logger.info("Closing transport...")
                     try:
                         self.ep.transportClose(self.transport)
                         # Give PJSIP a moment to process transport closure
                         time.sleep(0.1)
                     except pj.Error as e:
                         logger.warning(f"PJSIP Error closing transport: {e.info()} - proceeding with shutdown.")
                     finally:
                         self.transport = None


                # 4. Destroy Library (this cleans up everything else)
                logger.info("Destroying PJSIP Library...")
                self.ep.libDestroy()
                self.ep = None # Release reference
                logger.info("PJSIP Library Destroyed.")

            else:
                 logger.info("PJSIP already stopped or not initialized.")

        except pj.Error as e:
            logger.error(f"PJSIP Error during shutdown: {e.info()}")
        except Exception as e:
            logger.error(f"General Error during shutdown: {e}", exc_info=True)
        finally:
            # Ensure references are cleared
            self.ep = None
            self.acc = None
            self.transport = None
            # Clear global queues explicitly just in case
            playback_queues.clear()
            processing_signals.clear()
            transfer_requests.clear()
            logger.info("SIP Client Shutdown Complete.")


# --- Main Execution ---
if __name__ == "__main__":
    app = SipClientApp()
    app.start() # start() now contains the main loop and shutdown logic
