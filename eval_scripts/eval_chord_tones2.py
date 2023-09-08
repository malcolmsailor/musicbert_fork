from fairseq import checkpoint_utils, data, options, tasks


def main():
    # Parse command-line arguments for generation
    parser = options.get_generation_parser(default_task="musicbert_sequence_tagging")
    args = options.parse_args_and_arch(parser)

    # args.arch = "musicbert_base"  # type:ignore

    # Setup task
    task = tasks.setup_task(args)  # type:ignore

    # Load model
    print("| loading model from {}".format(args.restore_file))  # type:ignore
    models, _model_args = checkpoint_utils.load_model_ensemble(
        [args.restore_file],  # type:ignore
        task=task,
        # (Malcolm 2023-09-08) it seems to be necessary to override "_name", not sure if
        #   "arch" is necessary too. Should add command line args for this.
        arg_overrides={"_name": "musicbert_base", "arch": "musicbert_base"},
    )
    model = models[0]

    # TODO: (Malcolm 2023-09-08)
    # model.cuda()
    model.eval()

    eval_dataset = task.load_dataset("valid")

    batch = eval_dataset.collater([eval_dataset[0], eval_dataset[1]])
    logits, _ = model(
        **batch["net_input"],
        features_only=True,
        classification_head_name="sequence_tagging_head",
    )
    preds = logits.argmax(dim=-1)

    max_tokens = 24
    total_correct = 0
    total = 0
    token_length = 1
    split = "valid"
    i = 0
    for pred, target in zip(preds, batch["target"]):
        # Ignore padding/bos/eos
        valid_mask = target >= 0
        pred = pred[valid_mask]
        target = target[valid_mask]
        total += valid_mask.sum()
        total_correct += (pred == target).sum()

        #
        pred = pred[:max_tokens]
        target = target[:max_tokens]

        # We need to adjust for the specials at the beginning
        #   of the dictionary
        pred += task.label_dictionary.nspecial
        target += task.label_dictionary.nspecial

        target_tokens = task.label_dictionary.string(target)
        pred_tokens = task.label_dictionary.string(pred)
        target_tokens = [
            f"{x[:token_length]:<{token_length}}" for x in target_tokens.split()
        ]
        pred_tokens = [
            f"{x[:token_length]:<{token_length}}" for x in pred_tokens.split()
        ]
        target_tokens = " ".join(target_tokens)
        pred_tokens = " ".join(pred_tokens)

        print(f"{split} target     {i + 1}: ", target_tokens)
        print(f"{split} prediction {i + 1}: ", pred_tokens)
    print("Accuracy: ", total_correct / total)
    breakpoint()


if __name__ == "__main__":
    main()
