from datasets import load_dataset, Features, Value, concatenate_datasets
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException   
features = Features({
    "messages": [
        {
            "role": Value("string"),
            "content": Value("string"),
            "info": {
                "source": Value("string"),
                "reference_answer": Value("string"),
                "test_case": Value("string"),
                "think_content": Value("string"),
                "answer_content": Value("string")
            }
        }
    ]
})
# merge am_0.5M and am_0.9M subset
data1 = load_dataset('a-m-team/AM-DeepSeek-R1-Distilled-1.4M', 'am_0.5M', features=features)
data2 = load_dataset('a-m-team/AM-DeepSeek-R1-Distilled-1.4M', 'am_0.9M', features=features)
data = concatenate_datasets([data1['train'], data2['train']])
 
# Define a filter function to keep only English messages
def is_english(example):
    messages = example["messages"]
    # Assuming messages is a list of strings or dictionaries with text fields
    # Adjust based on your data structure
    question = messages[0]["content"]
    try:
        return detect(question) == "en"
    except LangDetectException:
        return False

# Apply filter to dataset
data_eng = data.filter(is_english, num_proc=128)

# You can push the dataset to the hub using push_to_hub
repo_id="YOUR_AMMATH_DATASET"
data_eng['train'].push_to_hub(
    repo_id=repo_id,  # The name of the repo on Hugging Face
    split="train",  # Specify the split (train, validation, etc.)
    private=False  # Set to True if you want the dataset to be private
)
