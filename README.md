
# SIP AI Assistant (Manufacturing Edition)

This project provides an automated SIP client that integrates with OpenAI's Whisper for speech-to-text, Google's Gemini for natural language processing, and gTTS for text-to-speech to create a manufacturing-focused AI assistant. It can access data from CSV files to provide estimates, look up contacts, and facilitate call transfers.

## Features

*   **Automatic Call Answering:** Automatically answers incoming SIP calls.
*   **Speech-to-Text:** Transcribes the caller's speech using OpenAI Whisper.
*   **Natural Language Processing:** Uses Google Gemini to understand the caller's intent and generate responses.
*   **Text-to-Speech:** Synthesizes Gemini's responses using gTTS.
*   **CSV Data Integration:** Accesses data from CSV files (company directory, cost sheet, lead times) to answer questions.
*   **Call Transfer:** Can transfer the call to a specified extension or SIP URI.
*   **Voice Activity Detection (VAD):** Uses `webrtcvad` for efficient audio processing, only transcribing audio when speech is detected.
*   **Conversation History:** Maintains a conversation history to provide context to Gemini.

## Prerequisites

*   **PJSIP/pjsua2:**  This is the most challenging part. You need to compile and install PJSIP with Python bindings (`pjsua2`).  Refer to the official PJSIP documentation ([https://www.pjsip.org/](https://www.pjsip.org/)) for your operating system.  `pip install pjsua2` will often *not* work directly; you typically need to build from source.
*   **Python 3.7+**
*   **API Keys:**
    *   OpenAI API key for Whisper.
    *   Google Gemini API key.
*   **SIP Account:** A SIP account with a provider.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   **Linux/macOS:**

        ```bash
        source venv/bin/activate
        ```

    *   **Windows:**

        ```bash
        venv\Scripts\activate
        ```

4.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure the application:**

    *   Create a `config.ini` file (see example below). Fill in your SIP credentials, API keys, and file paths.
    *   Or set environment variables if you choose (not described).

6.  **CSV Files:**

    *   Create the CSV files as specified (see format below).  These are not included in the repository and you'll need to create them yourself.

## Configuration (`config.ini`)

```ini
[SIP]
registrar = sip:your_sip_provider.com
username = your_username
password = your_password
realm = *

[API_KEYS]
openai_api_key = sk-YOUR_OPENAI_API_KEY
gemini_api_key = YOUR_GOOGLE_GEMINI_API_KEY

[AUDIO]
sample_rate = 16000

[VAD]
aggressiveness = 1
frame_ms = 30
silence_padding_ms = 300
speech_min_ms = 150

[CSV_PATHS]
directory = company_directory.csv
costs = cost_sheet.csv
lead_times = lead_time_sheet.csv

[TRANSFER]
default_target_uri = sip:sales@example.com

[PROMPTING]
system_prompt = """You are an AI assistant for a manufacturing company. Your goal is to provide helpful initial information to callers.
    - Be polite and professional.
    - You can answer questions about company contacts, basic part costs, and estimated lead times using your available tools.
    - Costs and lead times you provide are ESTIMATES ONLY. Always state this clearly.
    - To get a formal quote, the caller needs to provide detailed specifications (drawings, materials, tolerances, quantities). Explain this process.
    - If you look up information, summarize it naturally. Do not just dump raw data.
    - If asked for someone specific and you find them, offer to transfer the call. Use the format [TRANSFER:sip:target@domain.com] or [TRANSFER:sip:extension@domain.com] at the end of your response if a transfer is appropriate.
    - If the request is too complex, explain that you cannot provide an estimate and offer to transfer them to the appropriate department (e.g., Sales or Quotes team - check the directory tool) or explain how to submit a formal quote request.
    - Keep responses relatively concise for a phone call.
    - If you cannot find information using a tool, say so politely.
    """
```

## CSV File Formats (Examples)

These files are required for the application to function correctly.  These example contents are for demonstration only.

### `company_directory.csv`

```csv
Name,Department,Extension,Email
Alice Wonderland,Sales,101,alice@example.com
Bob The Builder,Engineering,202,bob@example.com
Charlie Chaplin,Support,303,charlie@example.com
David Copperfield,Quotes,404,david@example.com
```

*   **Name:** The employee's full name.
*   **Department:** The department the employee works in.
*   **Extension:** The employee's phone extension (optional, used for call transfer).
*   **Email:** The employee's email address (optional).

### `cost_sheet.csv`

```csv
PartNumber,Description,Material,CostPerUnit,Unit
PN-1001,Standard Widget,Steel,12.50,each
PN-1002,Large Widget,Aluminum,25.00,each
SVC-MACH,Machine Setup Fee,,150.00,job
SVC-ASSY,Assembly Labour,,"75.00",hour
```

*   **PartNumber:** The unique part number.
*   **Description:** A description of the part.
*   **Material:** The material the part is made from.
*   **CostPerUnit:** The cost per unit of the part.
*   **Unit:** The unit of measurement for the cost (e.g., "each", "hour", "job").

### `lead_time_sheet.csv`

```csv
ItemType,Operation,BaseDays,ComplexityFactor,Notes
Widget,Machining,5,1.0,Standard steel widget
Widget,Machining+Assembly,7,1.2,Requires assembly post-machining
Custom Part,Design,10,1.5,"Requires engineering design time, factor per complexity estimate"
Standard Order,Shipping,2,1.0,Domestic ground
Rush Order,Expedite Fee Factor,,1.5,"Applies to base days for rush"
```

*   **ItemType:**  The general type of item or service (e.g., "Widget", "Custom Part", "Standard Order").
*   **Operation:** A manufacturing or processing operation (e.g., "Machining", "Assembly", "Shipping").
*   **BaseDays:**  The base number of days for the operation or item.
*   **ComplexityFactor:** A factor to multiply the base days by, depending on the complexity of the job.
*   **Notes:**  Additional notes about the lead time.

## Running the Application

1.  Activate the virtual environment (if you haven't already).
2.  Run the main script:

    ```bash
    python sip_ai_assistant_v2.py
    ```

3.  Call the configured SIP number.

## Troubleshooting

*   **PJSIP Installation:** Double-check your PJSIP installation.  Ensure that `pjsua2.py` and the compiled `_pjsua2` library are correctly placed in your Python environment.  Review PJSIP build instructions for your OS.
*   **API Keys:** Verify that your API keys are correct in `config.ini`.
*   **Configuration:** Double-check all configuration values in `config.ini`.
*   **CSV Files:** Ensure that the CSV files exist in the specified paths and are formatted correctly.
*   **Logs:** Examine the application's logs for errors.

## Requirements

*   pjsua2
*   openai
*   google-generativeai
*   gTTS
*   pydub
*   SpeechRecognition
*   configparser
*   numpy
*   soundfile
*   webrtcvad-wheels
*   pandas
  
