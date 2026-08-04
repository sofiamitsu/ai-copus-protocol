from dotenv import load_dotenv
load_dotenv()

from validate.validator import compute_kappa

compute_kappa(
    csv_a="human_coding_lecture_001.csv",
    csv_b="output/pipeline_test/results.csv",
    output_dir="output/pipeline_test/validation",
    label_a="Human", label_b="AI",
)