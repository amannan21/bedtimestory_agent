## Setup Instructions

### Prerequisites
1. **OpenAI API Key Configuration**
   - Input your OpenAI API key when prompted

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. Running the Application
   
     a.) Streamlit Web Interface

   ```bash
      streamlit run chatUI.py

 
      b.) Command Line Interface
   ```bash
      python main.py
   '''

<img width="371" height="651" alt="Screenshot 2025-08-26 at 5 57 55 PM" src="https://github.com/user-attachments/assets/957ddb99-887b-459f-af89-c9e21c3d2afa" />




**Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:**   

If I had more time, I would have the similar system but I would have the explored diverse candidates at the story generation phase (for instance, pick 6 or so candidates with different temperatures and slightly different seeding instructions). Then I would have created one of these: 1. a new pairwise LLM Judge in charge of picking the best one of those candidates 2. Panel of LLm evaluators (as outlined here https://arxiv.org/pdf/2404.18796) 3.a compiler agent to take the best parts of each story and compile them into one (would have tested to determine which was better fit). The issue with relying on a solely single output judge is that scores drift, calibration can be fragile, and it can be gamed more easily by verbosity or keyword stuffing than a pairwise judge.

Alternatively, I would have also explored decoupling the revise agent into a few different agents (example: rhyme polisher, parental-controls, etc.) instead of just the single reviser model.

Over a longer period of 2 hours I would’ve also tried implementing automatic red-teaming using deepeval library for robustness.
