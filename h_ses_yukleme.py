# Import necessary libraries
import gradio as gr # Gradio for UI components
import google.generativeai as genai # Google Generative AI for processing
from api_read import GEMINI_API_KEY # Import the API key from an external file

# Initialize the API with the provided key
genai.configure(api_key=GEMINI_API_KEY)

# Define a function to process the uploaded audio file
def process_audio(audio_file, prompt, model_sel, lang_sel):
    # Create an instance of the selected AI model
    model = genai.GenerativeModel(model_sel)

    # Read the audio file in binary mode
    with open(audio_file, "rb") as f:
        audio_data = f.read()

    # Create a language-specific prompt for transcription
    lang_prompt = f"Ses dosyasını {lang_sel} dilinde transkript et."
    prompt = lang_prompt + prompt # Combine the language prompt with the user input

    # Send the requests to the model with both text and audio content
    response = model.generate_content(
        contents=[
            prompt, # Text prompt
            {
                "mime_type": "audio/mpeg", # Specify that the input is and audio file
                "data": audio_data # Attach the binary audio data
            }
        ]
    )

    # Return the transcribed text from the AI model
    return response.text

# Create a Gradio Blocks interface with a structured layout
with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    gr.Markdown("# Gemini Ses Dosyası Transkript Asistanı") # Title of the UI

    with gr.Row(): # Create a row to organize elements side by side
        with gr.Column(): # Left column for input elements
            audio_file = gr.Audio(type="filepath", visible=False) # Hidden audio input field
            attach_audio_file = gr.Button("Ses Dosyası Yükle") # Button to upload video

            # Dropdown menu to select the AI model
            model_sel = gr.Dropdown(
                choices=["gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-lite"],
                value="gemini-2.0-flash",
                label="Model seçin"
            )

            # Checkbox group for selecting one or multiple languages
            language_sel = gr.CheckboxGroup(
                choices=["Türkçe", "İngilizce", "Almanca", "İspanyolca","Rusça"],
                value="Türkçe",
                label = "Dil seçin"
            )

            prompt = gr.Textbox(label="Prompt") # Textbox for additional input
            submit_btn = gr.Button("Gönder") # Button to submit the request

        with gr.Column(): # Right column for output display
            output = gr.Textbox(label="Output") # Textbox to show transcribed text

    # Function to make the audio upload field visible when clicked
    def toggle_upload_audio():
        return gr.update(visible=True)

    # When the upload button is clicked, reveal the hidden audio input field
    attach_audio_file.click(
        toggle_upload_audio,
        [],
        [audio_file]
    )

    # When the submit button is clicked, process the audio and display output
    submit_btn.click(
        process_audio, # Function to process audio 
        [audio_file, prompt, model_sel, language_sel], # Inputs to the function
        [output] # Output to be displayed 
    )

# Run the Gradio app
if __name__ == "__main__":
    demo.launch(show_error=True)









