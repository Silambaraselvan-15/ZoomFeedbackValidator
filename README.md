# Zoom Feedback Validator

The **Zoom Feedback Validator** is a Gradio-based web application that leverages a fine-tuned BERT model to validate whether a feedback comment aligns with a selected reason. This tool is designed for classifying text-reason pairs in feedback data, such as Zoom meeting feedback or product reviews.

## Features

- Accepts a feedback comment and a selected reason as input.
- Utilizes a fine-tuned BERT classifier to determine if the feedback matches the reason.
- Provides an interactive and user-friendly web interface powered by **Gradio**.
- Outputs "Matched" or "Not Matched" based on the model's prediction.

## Demo

![App Screenshot](#)
*Replace this placeholder with a screenshot or GIF of the app interface.*

---

## Getting Started

### Prerequisites

Ensure the following dependencies are installed on your system:

- Python 3.7 or higher
- PyTorch
- Hugging Face Transformers
- Gradio
- pandas

### Installation

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/zoom-feedback-validator.git
   cd zoom-feedback-validator
   ```

2. **Install dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add model and data**

   - Place your fine-tuned model in a folder called `feedback_model/` (must be in Hugging Face format).
   - Add the evaluation dataset as `eval.csv` in the project directory.

     The CSV should follow this format (no header in original file):

     ```
     feedback_text,reason,label
     ```

4. **Run the app**

   Run the application using the following command:

   ```bash
   python app.py
   ```

   Or, if it's all in one script:

   ```bash
   python your_script_name.py
   ```

## File Structure

```
zoom-feedback-validator/
│
├── feedback_model/         # Folder containing fine-tuned BERT model
├── eval.csv                # Evaluation dataset with feedback and reasons
├── app.py                  # Python script containing Gradio app
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Example Usage

Enter:

- **Feedback Text:** "The breakout rooms were very helpful."
- **Reason:** "Breakout Rooms"

The app will return:

- **"Matched"** if the model thinks the reason fits the feedback.
- **"Not Matched"** otherwise.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Model fine-tuned using Hugging Face Transformers
- Interface built with Gradio
- Data format based on Zoom feedback use case

---

Let me know if you want help generating a sample `requirements.txt` or setting up deployment.
