# General Online Opinion Detector (GOOD)
With the variety of opinions on the internet, it can be extremely tedious and headache inducing to discover and compile all the information into a general idea. However, understanding the differences in viewpoints can facilitate opportunities for learning as well as personal growth and thoughtful opinion formation. Given how overwhelming websites like X and other news outputs can be, we sought to develop a solution to this prevalent issue. This problem can easily be realized by going on any of these websites, such as reddit, and typing in a keyword or topic. These topics can generally have hundreds of relevant posts with all sorts of different thoughts, overwhelming users instantly. We found this difficulty to be especially interesting and pressing, as the attention economy continues to be entirely consumed by social media. Billions of people globally spend hours every day on popular social media apps and websites, which can be considered detrimental to humanity’s progression. Many large corporations profit off of this, and continue to popularize their forms of media to gain more share in the global attention economy. Thus, methods of simplifying media have not developed. We sought to resolve this issue by using aspect based sentiment analysis (ABSA) with NLP. Although many models and language processing tools exist to resolve the internet’s abundance of information, we innovated by developing a system to search and compile text from various sources and apply ABSA to be more user friendly. This allows people without proficient technological expertise to still gain value from these models, which contributes to a growing diversity of online opinions. 

## Directory Structure

The repository is organized as follows:
```bash
GOOD/
├── BERT_testing/ # Scripts and notebooks related to BERT model testing
├── data/ # Contains datasets and data-related scripts
├── static/
│ └── images/ # Images used in the web interface
├── templates/ # HTML templates for the Flask web application
├── .gitignore # Specifies files to be ignored by Git
├── Final Report.pdf # Comprehensive report detailing the project
├── GOOD_poster.pptx.pdf # Poster presentation of the project
├── Project Presentations 5.pdf # Slide deck for project presentation
├── README.md # This README file
├── app.py # Main Flask application script
├── data.pkl # Pickled data file used by the application
├── instr.py # Script containing instructions or helper functions
├── main.py # Entry point for data processing and analysis
├── my_list.json # JSON file containing a list of topics or data
```

---

## Installation

To set up and run the GOOD application locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/williamcoryell/GOOD.git
cd GOOD
```

2. Create a Virtual Environment (Optional)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies

Ensure you have pip installed, then run:
```bash
pip install -r requirements.txt
```
If requirements.txt is not present, manually install needed packages such as Flask, transformers, torch, etc.

4. Download or Prepare Data
Make sure data.pkl and my_list.json are in the root directory. If not, generate or obtain these files according to project needs.

Usage
1. Run the Application
Start the Flask web application:

```bash
python app.py
```

By default, visit the app at http://127.0.0.1:5000/ in your web browser.

2. Use the Interface
Enter a topic or keyword of interest.
View the aggregated opinions, sentiment analysis, and visualizations.


