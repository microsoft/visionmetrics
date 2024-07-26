from visionmetrics.caption.azure_openai_model_eval_base import AzureOpenAITextModelCategoricalEvalBase


class AzureOpenAITextModelCategoricalScore(AzureOpenAITextModelCategoricalEvalBase):
    """
    Prompt-tuned evaluator that uses single-turn text-only calls to an Azure OpenAI model for evaluation and returns
    standard categorical metrics (precision, recall, F1, accuracy) based on the scores returned.
    """

    def __init__(self):
        self.system_message = """You are an expert evaluator of AI models. Follow the steps below:
1. Carefully compare the prediction generated by the model against the correct ground truth value and determine how much they differ. Equally consider important semantic components of the response and details and numbers. Note that in the ground truth values, <|OR|> indicates that a match is only required with any of the values between the <|OR|> separator.
2. Assign a correctness score for the predicted value. Only provide a numeric score between 0.0 and 1.0. Do not provide rationale or other text; ONLY write the score.

Example:
Ground truth: The multi-axes chart titled "Healthcare and Health Analysis: Cost, Coverage, and Premiums" presents data across several healthcare categories. Private Health Insurance leads in cost at $98,800 and covers 9,502 patients with an average premium of $3,760 and 6 insurers. Outpatient Care costs $70,000, covers 8,502 patients, with $4,500 average premiums and 4 insurers. Inpatient Care shows $80,000 in costs, 6,200 patients, a $3,500 average premium, and 5 insurers. Mental Health Care has a $60,000 cost, covers 4,200 patients, and has $4,000 average premiums with the highest number of insurers at 7. Dental Care has a $54,000 cost for 3,000 patients covered, a $3,000 average premium, and 9 insurers. Vision Care, at $18,000 cost, covers the fewest patients, 1,500, with a $2,500 average premium and 3 insurers. Lastly, Prescription Drugs cost $20,000, cover 5,000 patients, and have a $2,600 average premium with 4 insurers.
Prediction: The bar chart visualizes the cost in dollars across different healthcare categories, along with the number of patients covered, average premiums, and number of insurers. The healthcare categories include Private Health Insurance with 9,500 patients, Outpatient Care with 8,500 patients, Inpatient Care with 6,000 patients, Mental Health Care with 4,000 patients, Dental Care with 3,000 patients, Vision Care with 1,000 patients, and Prescription Drugs with 5,000 patients. Costs are represented by blue bars, patients covered by a blue line, average premiums by pink markers connected with lines, and insurers indicated on the secondary y-axis.
Score: 0.6

Ground truth: 3 dogs
Prediction: 4 dogs
Score: 0.2

Ground truth: The Hunger Games: Catching Fire
Prediction: The Hunger Games
Score: 0.7

Ground truth: 
Prediction: 20 pounds
Score: 0.0"""
        self.prompt_template = """Ground truth: <|target|>
Prediction: <|prediction|>
Score: """
        super().__init__(endpoint="https://customvision-dev-aoai.openai.azure.com/",
                         deployment_name="gpt4o-004",
                         system_message=self.system_message,
                         prompt_template=self.prompt_template,
                         temperature=0.0,
                         max_tokens=50,
                         positive_threshold=0.5,
                         negative_value='')
        