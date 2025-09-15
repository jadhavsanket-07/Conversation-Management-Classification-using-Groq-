# 🗣️ Conversation Management & Classification using Groq API  
**An advanced Python implementation demonstrating conversational data management, summarization, and structured chat classification using Groq’s OpenAI-compatible API — without using any frameworks.**

---

## 🎯 Project Objective  
This project develops and showcases two core functionalities leveraging Groq APIs compatible with OpenAI's SDK:  

1. **Conversation History Management with Summarization**  
2. **JSON Schema Classification & Structured Information Extraction**  

It validates handling dynamic chat interactions, generating concise summaries, truncating histories, and accurately extracting user details from text conversations all implemented with standard Python libraries for clarity and control.

---

## 🛠️ Features & Capabilities

### Task 1: Managing Conversation History with Summarization  
- **Running History Maintenance:** Automatically track the full sequence of user assistant exchanges with timestamps and role annotations.  
- **Flexible Truncation:** Supports limiting conversation context by:  
  - Last *n* messages
  - Maximum character count  
  - Maximum word count  
- **Periodic Summarization:**  
  - Auto trigger summarization after every *k*-th conversation run
  - Optionally replace detailed history with summarized content to keep it concise  
- **Customizable Summarization:** Summaries can be tailored using instructions for desired brevity and structure.  
- **Demonstration Coverage:** Feeds multiple sample conversations showcasing truncation modes and periodic summarization effects.  

### Task 2: JSON Schema Classification & Information Extraction  
- **Robust JSON Schema:** Designed for extracting five essential user details from chats:  
  - Name  
  - Email  
  - Phone number  
  - Location  
  - Age  
- **OpenAI Function Calling Integration:** Uses Groq API’s OpenAI-compatible function call feature for structured output extraction.  
- **Extraction Validation:**  
  - Regex based validation for email and phone formats  
  - Age validation and retry mechanism for correct integer parsing  
- **Sample Chats Processing:** Parses multiple example conversations and demonstrates validation and extraction accuracy.  
- **Persistence:** Saves the extracted and validated user information into an SQLite database for record keeping.

---

## 🚀 Getting Started

### Prerequisites  
- Python  
- Internet connection for API requests  
- Groq API key with access to the OpenAI compatible endpoint
  
## 🚀 Installation & Running
1. Clone the repository
2. Install dependencies: `pip install requests`
3. Add your Groq API key in the notebook
4. Run the Colab notebook or Python script
