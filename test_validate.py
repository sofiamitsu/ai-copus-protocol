from dotenv import load_dotenv
load_dotenv()

from validate.validator import compute_kappa

compute_kappa(
    human_csv="human_coding_lecture_001.csv",
    ai_csv="output/pipeline_test/results.csv",
    output_dir="output/pipeline_test/validation"
)