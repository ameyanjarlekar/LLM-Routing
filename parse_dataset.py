import json
from pathlib import Path


def extract_quality_vectors(
    input_file: str, output_file: str = "llm_routing_poc_data/parsed_quality_vectors.txt"
):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_written = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:

        # Header for the text file
        outfile.write(f"{'Query Num/ID':<15} | {'Model Quality Vector / Scores'}\n")
        outfile.write("-" * 65 + "\n")

        for index, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract query identifier or fallback to line index
                query_id = (
                    data.get("query_id")
                    or data.get("id")
                    or data.get("query_num")
                    or f"Query #{index}"
                )

                # Extract quality vector across common schema keys
                quality_vector = (
                    data.get("quality_vector")
                    or data.get("model_scores")
                    or data.get("scores")
                    or data.get("quality")
                    or data.get("models")
                )

                if quality_vector is None and "eval" in data:
                    quality_vector = data["eval"].get("scores")

                # Format quality vector as JSON string if it's a dict or list
                if isinstance(quality_vector, (dict, list)):
                    formatted_vector = json.dumps(quality_vector)
                else:
                    formatted_vector = str(quality_vector)

                # Write line to file
                outfile.write(f"{str(query_id):<15} | {formatted_vector}\n")
                records_written += 1

            except json.JSONDecodeError:
                print(f"Skipping malformed line {index}")

    print(f"Successfully processed {records_written} entries.")
    print(f"Output saved to: {output_path.resolve()}")


if __name__ == "__main__":
    input_path = "llm_routing_poc_data/routing_dataset.jsonl"
    output_path = "llm_routing_poc_data/parsed_quality_vectors.txt"

    extract_quality_vectors(input_path, output_path)
