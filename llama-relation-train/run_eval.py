import json
import fire
import evaluate
import importlib
import os


def import_inst_format_fn(fn_name):
    try:
        inst_util_module = "eval_util"
        module = importlib.import_module(inst_util_module)
        try:
            fn = getattr(module, fn_name)
            print(
                f"Function '{fn_name}' from module '{inst_util_module}' successfully imported."
            )
            return fn
        except AttributeError:
            print(f"Function '{fn_name}' not found in module '{inst_util_module}'.")
    except ImportError:
        print(f"Could not import module '{inst_util_module}'.")


def main(
    prediction_path,
    metric,
    gold_field="label",
    pred_field="output_prediction",
    post_processing_fn=None,
    format="json",
):
    if post_processing_fn is not None:
        post_processing = import_inst_format_fn(post_processing_fn)
    else:
        post_processing = lambda x: x
    with open(prediction_path) as f:
        if format == "json":
            prediction_data = json.load(f)
        elif format == "jsonl":
            prediction_data = [json.loads(line) for line in f]
    eval_metric = evaluate.load(metric)

    preds = [post_processing(x[pred_field]) for x in prediction_data]
    refs = [x[gold_field] for x in prediction_data]

    results = eval_metric.compute(references=refs, predictions=preds)

    output_path = os.path.join(os.path.dirname(prediction_path), "eval_result.json")
    print("*** Evaluation Result ***")
    print(json.dumps(results, indent=4))
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
