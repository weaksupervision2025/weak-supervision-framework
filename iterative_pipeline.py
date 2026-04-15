import argparse
import json
from pathlib import Path
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm
from loguru import logger
from snorkel.labeling import LFAnalysis
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from catboost import CatBoostClassifier, Pool

from src.llm_client import LLMQueryClient
from src.criteria_generator import CriteriaGenerator
from src.classifier import DialogueCriteriaClassifier
from src.snorkel_trainer import SnorkelTrainer
from src.dawid_scene_trainer import DawidSkeneTrainer
from src.majority_vote_trainer import MajorityVoteTrainer
from generate_criteria import read_criteria
from classify_criteria import run_parallel_requests
from prepare_datasets import prepare_banking77_dataset, prepare_chemprot_dataset, prepare_claude9_dataset, prepare_tarif_dataset
from src.train_bert import train_bert
import config

import os
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES_BERT


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset_df(dataset_name: str, split: str, base_dir: str = "data") -> pd.DataFrame:
    json_path = os.path.join(base_dir, dataset_name, "source", f"{split}.json")

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Split file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    records = []
    for idx, entry in raw.items():
        text = entry.get("data", {}).get("text")
        label = entry.get("label")
        weak_labels = entry.get("weak_labels", None)
        records.append({"text": text, "label": label, "weak_labels": weak_labels})

    df = pd.DataFrame(records)
    return df


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def read_criteria_file(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    descriptions: dict[str, str] = {}
    classes: dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            criterion_name = obj["criterion"]
            descriptions[criterion_name] = obj["description"]

            try:
                classes[criterion_name] = int(obj["class"])
            except Exception as e:
                classes[criterion_name] = obj["class"]

    return descriptions, classes


def filter_lfs(
    pred_df: pd.DataFrame,
    trainer: SnorkelTrainer,
    threshold: float,
    metrics_dir: Path,
) -> list[str]:
    L = trainer.applier.apply(pred_df)
    scores = LFAnalysis(L, trainer.lfs).lf_summary(pred_df["label"].values)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scores_path = metrics_dir / "lf_scores.csv"
    scores.to_csv(scores_path)
    logger.info(f"Saved LF scores to {scores_path}")
    good_lfs = scores[scores["Emp. Acc."] > threshold].index.tolist()
    logger.info(f"Filtered to {len(good_lfs)} labeling functions")
    return good_lfs


def compute_metrics(y_true, y_pred, average="macro"):
    return {
        "f1": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "acc": accuracy_score(y_true, y_pred)
    }


def load_classified(path: Path) -> pd.DataFrame:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            row = {"index": obj["index"], "text": obj["text"]} | obj["labels"]
            data.append(row)
    return pd.DataFrame(data)


def classify_texts(llm_client, texts: list[str], criteria: dict[str, str], output: Path, workers: int, domain_info: str | None = None) -> pd.DataFrame:
    if output.exists():
        logger.info(f"Using existing classification from {output}")
    classifier = DialogueCriteriaClassifier(llm_client, criteria, config.CLASSIFY_PROMPT_FILE, config.LLM_FOR_CLASSIFY)
    start = 0
    if output.exists():
        with open(output, "r", encoding="utf-8") as f:
            start = sum(1 for _ in f)
        if start >= len(texts):
            return load_classified(output)
        texts = texts[start:]
    run_parallel_requests(texts, classifier, str(output), workers, start_idx=start, domain_info=domain_info)
    return load_classified(output)


def run_parallel_generation(
    generator: CriteriaGenerator,
    dataset: str,
    label_groups: dict[str, list[str]],
    label_groups_already_correct: dict[str, list[str]],
    existing: dict[str, str] | None,
    num_workers: int,
    domain_info: str | None = None,
    number_of_criteria: int = 5,
) -> list[dict[str, str]]:
    logger.info(
        f"Generating criteria for {len(label_groups)} groups with {num_workers} workers..."
    )
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for label, true_texts in tqdm(label_groups.items(), total=len(label_groups)):
            false_texts = []
            already_correct_texts = []
            for other_label, other_texts in label_groups.items():
                if other_label != label:
                    false_texts.extend(other_texts)
            for other_label, other_texts in label_groups_already_correct.items():
                if other_label != label:
                    false_texts.extend(other_texts)
                if other_label == label:
                    already_correct_texts.extend(other_texts)
            futures = {
                executor.submit(
                    generator.get_new_criteria,
                    dataset,
                    true_texts,
                    false_texts,
                    already_correct_texts,
                    label,
                    number_of_criteria=number_of_criteria,
                    existing_criteria=existing,
                    domain_info=domain_info,
                ): label
            }
            for future in futures:
                label = futures[future]
                try:
                    res = future.result()
                except Exception as e:  # pragma: no cover - logging only
                    logger.error(
                        f"Error generating criteria for label {label}: {repr(e)}"
                    )
                    res = []
                results.extend(res)
    return results


def run_iteration(args, iteration: int, exp_id: str, error_texts: list[dict[str, str]] | None = None,
                  correct_texts: list[dict[str, str]] | None = None):
    iter_dir = Path(args.output_dir) / args.dataset / exp_id / f"iter_{iteration}"
    # Prepare directory structure for this iteration
    ensure_dir(iter_dir / "weak_labels")
    ensure_dir(iter_dir / "models")
    ensure_dir(iter_dir / "metrics")
    ensure_dir(iter_dir / "classified")

    # Load dataset splits
    train_df = load_dataset_df(args.dataset, "train")
    test_df = load_dataset_df(args.dataset, "test")
    dev_df = load_dataset_df(args.dataset, "dev")
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Test size: {len(test_df)}")
    logger.info(f"Dev size: {len(dev_df)}")
    logger.info(f"Wrong size: {len(error_texts) if error_texts else 0}")
    logger.info(f"Correct size: {len(correct_texts) if correct_texts else 0}")

    domain_info = json.load(open(os.path.join("data", args.dataset, "domain_info.json"), "r")).get("domain_info", None)
    if domain_info == "":
        domain_info = None

    # Train is unlabeled
    train_df.drop("label", axis=1, inplace=True)

    llm_client = LLMQueryClient(config.API_BASE_URL, config.API_KEY, config.VLLM_BASE_URL, config.VLLM_KEY)

    # Initialize criteria generator
    generator = CriteriaGenerator(
        llm_client,
        config.GENERATION_PROMPT_FILE,
        config.DEDUPLICATION_PROMPT_FILE,
        model=config.LLM_FOR_GENERATION
    )

    if error_texts:
        texts = [sample["text"] for sample in error_texts]
        labels = [str(sample["label"]) for sample in error_texts]
        texts_already_correct = [sample["text"] for sample in correct_texts]
        labels_already_correct = [str(sample["label"]) for sample in correct_texts]
    else:
        texts = dev_df["text"].tolist()
        labels = dev_df["label"].tolist()
        texts_already_correct = []
        labels_already_correct = []


    criteria_path = iter_dir / "criteria.jsonl"
    prev_path = (
        Path(args.output_dir)
        / args.dataset
        / exp_id
        / f"iter_{iteration-1}"
        / "criteria.jsonl"
    ) if iteration > 0 else None
    existing = read_criteria(prev_path) if prev_path and prev_path.exists() else []

    if criteria_path.exists():
        # Criteria already calculated in a previous run
        logger.info(f"Loading existing criteria from {criteria_path}")
        criteria_descriptions, criteria_classes = read_criteria_file(criteria_path)
    else:
        label_groups: dict[str, list[str]] = {}
        for t, l in zip(texts, labels):
            label_groups.setdefault(l, []).append(t)
        label_groups_already_correct: dict[str, list[str]] = {}
        for t, l in zip(texts_already_correct, labels_already_correct):
            label_groups_already_correct.setdefault(l, []).append(t)

        existing_dict = (
            {c["criterion"]: c["description"] for c in existing} if existing else None
        )
        new_criteria = []
        for _ in range(config.GENARATION_ITERS):
            new_criteria.extend(run_parallel_generation(
                generator,
                args.dataset,
                label_groups,
                label_groups_already_correct,
                existing_dict,
                args.num_workers,
                domain_info=domain_info,
                number_of_criteria=(config.NUMBER_OF_CRITERIA_PER_CLASS if iteration == 0 else 1)
            ))

        logger.info(f"Generated {len(new_criteria)} new criteria!")

        if config.DO_DEDUP:
            if existing:
                if (len(new_criteria) + len(existing)) > config.MAX_CRITERIA_FOR_DEDUP:
                    new_criteria = random.sample(new_criteria, config.MAX_CRITERIA_FOR_DEDUP - len(existing))
                # final_criteria = generator.deduplicate_new_criteria(existing, new_criteria)
                final_criteria = existing + new_criteria
            else:
                if len(new_criteria) > config.MAX_CRITERIA_FOR_DEDUP:
                    new_criteria = random.sample(new_criteria, config.MAX_CRITERIA_FOR_DEDUP)
                # final_criteria = generator.deduplicate_new_criteria([], new_criteria)
                final_criteria = new_criteria
        else:
            final_criteria = existing + new_criteria

        # final_criteria_desc = {
        #     "activate_my_card": [
        #         " activat+card",
        #         "new+card+start"
        #     ],
        #     "age_limit": [
        #         "age limit",
        #         "child",
        #         "my son",
        #         "my daughter"
        #     ],
        #     "apple_pay_or_google_pay": [
        #         "apple",
        #         "google"
        #     ],
        #     "atm_support": [
        #         "atm+can i",
        #         "atm+where",
        #         "atm+how"
        #     ],
        #     "automatic_top_up": [
        #         "auto+top",
        #         "auto+add money"
        #     ],
        #     "balance_not_updated_after_bank_transfer": [
        #         " change+transfer",
        #         " updat+transfer",
        #         " show+transfer"
        #     ],
        #     "balance_not_updated_after_cheque_or_cash_deposit": [
        #         " change+ cheque",
        #         " updat+ cheque",
        #         " show+ cheque",
        #         " change+ deposit",
        #         " updat+ deposit",
        #         " show+ deposit"
        #     ],
        #     "beneficiary_not_allowed": [
        #         "beneficiary+n't",
        #         "beneficiary+not"
        #     ],
        #     "cancel_transfer": [
        #         "cancel+transfer",
        #         " stop+transfer"
        #     ],
        #     "card_about_to_expire": [
        #         "card+expir"
        #     ],
        #     "card_acceptance": [
        #         "card+accept",
        #         "card+ can+ use"
        #     ],
        #     "card_arrival": [
        #         "card+arriv",
        #         "card+come in"
        #     ],
        #     "card_delivery_estimate": [
        #         "how+card+deliv",
        #         "when+card+deliv"
        #     ],
        #     "card_linking": [
        #         "card+link",
        #         "card+connect",
        #         "card+reactivate"
        #     ],
        #     "card_not_working": [
        #         "card+not+work",
        #         "card+n't+work",
        #         "card+fail"
        #     ],
        #     "card_payment_fee_charged": [
        #         "card+extra+ fee",
        #         "card+extra+charge"
        #     ],
        #     "card_payment_not_recognised": [
        #         "pay+recogni+n't",
        #         "pay+recogni+not",
        #         "pay+unfamiliar"
        #     ],
        #     "card_payment_wrong_exchange_rate": [
        #         "exchange+rate+wrong",
        #         "exchange+rate+correct",
        #         "exchange+rate+ bad"
        #     ],
        #     "card_swallowed": [
        #         "card+swallow",
        #         "card+ ate",
        #         "card+ eat"
        #     ],
        #     "cash_withdrawal_charge": [
        #         "withdr+charge",
        #         "withdr+ fee",
        #         "atm+ fee",
        #         "atm+charge"
        #     ],
        #     "cash_withdrawal_not_recognised": [
        #         "withdr+recogni+n't",
        #         "withdr+recogni+not",
        #         "withdr+unfamiliar"
        #     ],
        #     "change_pin": [
        #         "change+ pin",
        #         "new+ pin"
        #     ],
        #     "compromised_card": [
        #         "card+someone+ use",
        #         "card+fraud",
        #         "card+someone+ using"
        #     ],
        #     "contactless_not_working": [
        #         "contactless"
        #     ],
        #     "country_support": [
        #         "countr+support",
        #         "countr+can+ use",
        #         "countr+available",
        #         "europe+can+ use",
        #         "europe+available"
        #     ],
        #     "declined_card_payment": [
        #         "pay+decline",
        #         "pay+n't+accept",
        #         "pay+not+accept"
        #     ],
        #     "declined_cash_withdrawal": [
        #         "withdr+decline"
        #     ],
        #     "declined_transfer": [
        #         "transfer+decline",
        #         "transfer+n't+accept",
        #         "transfer+not+accept"
        #     ],
        #     "direct_debit_payment_not_recognised": [
        #         "direct+debit",
        #         "debit+unauth",
        #         "debit+n't",
        #         "debit+not"
        #     ],
        #     "disposable_card_limits": [
        #         "disposable+limit",
        #         "disposable+max"
        #     ],
        #     "edit_personal_details": [
        #         "edit+personal+detail",
        #         "change+name",
        #         "change+personal+detail"
        #     ],
        #     "exchange_charge": [
        #         "exchange+discount",
        #         "exchange+ fee",
        #         "exchange+charge"
        #     ],
        #     "exchange_rate": [
        #         "exchange+ rate"
        #     ],
        #     "exchange_via_app": [
        #         "exchange+using+app",
        #         "exchange+with+app"
        #     ],
        #     "extra_charge_on_statement": [
        #         "extra+charge+statement",
        #         "extra+ fee+statement",
        #         "additional+charge+statement",
        #         "additional+fee+statement"
        #     ],
        #     "failed_transfer": [
        #         "transfer+fail",
        #         "transfer+not+complete"
        #     ],
        #     "fiat_currency_support": [
        #         "fiat+currenc"
        #     ],
        #     "get_disposable_virtual_card": [
        #         "disposable+virtual"
        #     ],
        #     "get_physical_card": [
        #         "what+ pin",
        #         "where+ pin",
        #         "when+ pin"
        #     ],
        #     "getting_spare_card": [
        #         "spare+card+get",
        #         "additional+card",
        #         "second+card"
        #     ],
        #     "getting_virtual_card": [
        #         "virtual+card+receive",
        #         "virtual+card+get",
        #         "virtual+card+order"
        #     ],
        #     "lost_or_stolen_card": [
        #         "lost+card",
        #         "stole+card",
        #         "find+card"
        #     ],
        #     "lost_or_stolen_phone": [
        #         "lost+phone",
        #         "stole+phone",
        #         "find+phone"
        #     ],
        #     "order_physical_card": [
        #         "physical+card+receiv",
        #         "physical+card+get",
        #         "physical+card+order"
        #     ],
        #     "passcode_forgotten": [
        #         "passcode+forgot",
        #         "passcode+reset",
        #         "password+forgot",
        #         "password+reset"
        #     ],
        #     "pending_card_payment": [
        #         "pay+pend",
        #         "pay+stuck",
        #         "pay+n't+available"
        #     ],
        #     "pending_cash_withdrawal": [
        #         "withdr+pend",
        #         "withdr+wait"
        #     ],
        #     "pending_top_up": [
        #         "top+up+pend",
        #         "top+up+n't+available",
        #         "top+up+not+available"
        #     ],
        #     "pending_transfer": [
        #         "transfer+pend",
        #         "transfer+stuck",
        #         "transfer+not+available"
        #     ],
        #     "pin_blocked": [
        #         " pin+ lock",
        #         " pin+block",
        #         " pin+wrong",
        #         " pin+correct"
        #     ],
        #     "receiving_money": [
        #         "receiv+money",
        #         "get+money",
        #         "deposit+money"
        #     ],
        #     "Refund_not_showing_up": [
        #         "refund+n't+show+ up",
        #         "refund+not+show+ up",
        #         "refund+too+long"
        #     ],
        #     "request_refund": [
        #         "refund+how",
        #         "refund+help",
        #         "refund+want",
        #         "refund+policy",
        #         "refund+need"
        #     ],
        #     "reverted_card_payment?": [
        #         "pay+revert",
        #         "pay+return",
        #         "pay+cancel"
        #     ],
        #     "supported_cards_and_currencies": [
        #         "support+currenc",
        #         "support+card",
        #         "american+express"
        #     ],
        #     "terminate_account": [
        #         "close +my+account",
        #         "delete+my+account",
        #         "terminate+my+account"
        #     ],
        #     "top_up_by_bank_transfer_charge": [
        #         "swift",
        #         "top+up+transfer",
        #         "top+up+bank"
        #     ],
        #     "top_up_by_card_charge": [
        #         "card+charge+top+up",
        #         "card+charge+add+money",
        #         "card+fee+top+up"
        #     ],
        #     "top_up_by_cash_or_cheque": [
        #         "top+up+cash",
        #         "top+up+check",
        #         "top+up+cheque"
        #     ],
        #     "top_up_failed": [
        #         "fail+top+up",
        #         "top+up+n't+work",
        #         "top+up+not+work"
        #     ],
        #     "top_up_limits": [
        #         "top+up+limit",
        #         "top+up+max"
        #     ],
        #     "top_up_reverted": [
        #         "top+up+revert",
        #         "top+up+cancel"
        #     ],
        #     "topping_up_by_card": [
        #         "top+up+card"
        #     ],
        #     "transaction_charged_twice": [
        #         "double+charge",
        #         "charge+twice",
        #         "duplicate",
        #         "more+once"
        #     ],
        #     "transfer_fee_charged": [
        #         "transfer+ charge",
        #         "transfer+ fee"
        #     ],
        #     "transfer_into_account": [
        #         "transfer+my+account",
        #         "transfer+into"
        #     ],
        #     "transfer_not_received_by_recipient": [
        #         "transfer+n't+arrive",
        #         "transfer+not+arrive",
        #         "transfer+n't+receive",
        #         "transfer+not+receive"
        #     ],
        #     "transfer_timing": [
        #         "how long+transfer",
        #         "transfer+time"
        #     ],
        #     "unable_to_verify_identity": [
        #         "identity+verif+not",
        #         "identity+verif+n't"
        #     ],
        #     "verify_source_of_funds": [
        #         "verif+fund",
        #         "where+money+from",
        #         "source+fund",
        #         "source+money"
        #     ],
        #     "verify_top_up": [
        #         "verif+top+up"
        #     ],
        #     "virtual_card_not_working": [
        #         "virtual+card+not+work",
        #         "virtual+card+n't+work",
        #         "virtual+card+reject"
        #     ],
        #     "visa_or_mastercard": [
        #         "visa",
        #         "mastercard"
        #     ],
        #     "verify_my_identity": [
        #         " id+check",
        #         "what+ id"
        #     ],
        #     "why_verify_identity": [
        #         "why+name",
        #         "why+ id"
        #     ],
        #     "wrong_amount_of_cash_received": [
        #         "atm+wrong+amount",
        #         "withdr+correct+amount",
        #         "amount+not+withdr",
        #         "amount+n't+withdr"
        #     ],
        #     "wrong_exchange_rate_for_cash_withdrawal": [
        #         "withdr+exchange+ rate",
        #         "withdr+holida",
        #         "withdr+travel"
        #     ]
        # }
        #
        # final_criteria_labels = {
        #     "0": "activate_my_card",
        #     "1": "age_limit",
        #     "2": "apple_pay_or_google_pay",
        #     "3": "atm_support",
        #     "4": "automatic_top_up",
        #     "5": "balance_not_updated_after_bank_transfer",
        #     "6": "balance_not_updated_after_cheque_or_cash_deposit",
        #     "7": "beneficiary_not_allowed",
        #     "8": "cancel_transfer",
        #     "9": "card_about_to_expire",
        #     "10": "card_acceptance",
        #     "11": "card_arrival",
        #     "12": "card_delivery_estimate",
        #     "13": "card_linking",
        #     "14": "card_not_working",
        #     "15": "card_payment_fee_charged",
        #     "16": "card_payment_not_recognised",
        #     "17": "card_payment_wrong_exchange_rate",
        #     "18": "card_swallowed",
        #     "19": "cash_withdrawal_charge",
        #     "20": "cash_withdrawal_not_recognised",
        #     "21": "change_pin",
        #     "22": "compromised_card",
        #     "23": "contactless_not_working",
        #     "24": "country_support",
        #     "25": "declined_card_payment",
        #     "26": "declined_cash_withdrawal",
        #     "27": "declined_transfer",
        #     "28": "direct_debit_payment_not_recognised",
        #     "29": "disposable_card_limits",
        #     "30": "edit_personal_details",
        #     "31": "exchange_charge",
        #     "32": "exchange_rate",
        #     "33": "exchange_via_app",
        #     "34": "extra_charge_on_statement",
        #     "35": "failed_transfer",
        #     "36": "fiat_currency_support",
        #     "37": "get_disposable_virtual_card",
        #     "38": "get_physical_card",
        #     "39": "getting_spare_card",
        #     "40": "getting_virtual_card",
        #     "41": "lost_or_stolen_card",
        #     "42": "lost_or_stolen_phone",
        #     "43": "order_physical_card",
        #     "44": "passcode_forgotten",
        #     "45": "pending_card_payment",
        #     "46": "pending_cash_withdrawal",
        #     "47": "pending_top_up",
        #     "48": "pending_transfer",
        #     "49": "pin_blocked",
        #     "50": "receiving_money",
        #     "51": "Refund_not_showing_up",
        #     "52": "request_refund",
        #     "53": "reverted_card_payment?",
        #     "54": "supported_cards_and_currencies",
        #     "55": "terminate_account",
        #     "56": "top_up_by_bank_transfer_charge",
        #     "57": "top_up_by_card_charge",
        #     "58": "top_up_by_cash_or_cheque",
        #     "59": "top_up_failed",
        #     "60": "top_up_limits",
        #     "61": "top_up_reverted",
        #     "62": "topping_up_by_card",
        #     "63": "transaction_charged_twice",
        #     "64": "transfer_fee_charged",
        #     "65": "transfer_into_account",
        #     "66": "transfer_not_received_by_recipient",
        #     "67": "transfer_timing",
        #     "68": "unable_to_verify_identity",
        #     "69": "verify_my_identity",
        #     "70": "verify_source_of_funds",
        #     "71": "verify_top_up",
        #     "72": "virtual_card_not_working",
        #     "73": "visa_or_mastercard",
        #     "74": "why_verify_identity",
        #     "75": "wrong_amount_of_cash_received",
        #     "76": "wrong_exchange_rate_for_cash_withdrawal"
        # }
        #
        # final_criteria_labels_r = {}
        # for k, v in final_criteria_labels.items():
        #     final_criteria_labels_r[v] = k
        #
        # final_criteria = []
        # for k, v in final_criteria_desc.items():
        #     for i, vv in enumerate(v):
        #         final_criteria.append({
        #             "criterion": k + str(i + 1),
        #             "description": vv,
        #             "class": final_criteria_labels_r[k]
        #         })


        logger.info(f"Writing {len(final_criteria)} criteria to {criteria_path}")
        with open(criteria_path, "w", encoding="utf-8") as f:
            for item in final_criteria:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        criteria_descriptions, criteria_classes = read_criteria_file(criteria_path)
    classes = sorted(dev_df["label"].unique().tolist())
    trainer_full = SnorkelTrainer(criteria_descriptions, criteria_classes, classes)

    dev_output = iter_dir / "classified" / "dev.jsonl"
    if dev_output.exists():
        logger.info(f"Loading existing classification from {dev_output}")
        dev_pred_df = load_classified(dev_output)
    else:
        dev_pred_df = classify_texts(
            llm_client,
            dev_df["text"].tolist(),
            criteria_descriptions,
            dev_output,
            args.num_workers,
            domain_info,
        )

    dev_pred_df = dev_pred_df.sort_values(by="index")
    dev_pred_df = dev_pred_df.set_index('index').reset_index()
    dev_pred_df["label"] = dev_df["label"].map(lambda x: trainer_full.class_to_index[x])

    good_lfs = filter_lfs(
        dev_pred_df,
        trainer_full,
        args.accuracy_threshold,
        iter_dir / "metrics",
    )
    filtered_criteria_descriptions = {
        k: v for k, v in criteria_descriptions.items() if k in good_lfs
    }
    filtered_criteria_classes = {
        k: v for k, v in criteria_classes.items() if k in good_lfs
    }
    with open(iter_dir / "filtered_lfs.json", "w", encoding="utf-8") as f:
        json.dump(filtered_criteria_descriptions, f, ensure_ascii=False, indent=2)

    # Initialize trainer with filtered labeling functions
    trainer = SnorkelTrainer(
        filtered_criteria_descriptions,
        filtered_criteria_classes,
        classes,
    )

    # classify train and test with filtered criteria
    train_output = iter_dir / "classified" / "train.jsonl"
    test_output = iter_dir / "classified" / "test.jsonl"
    if train_output.exists():
        logger.info(f"Loading existing classification from {train_output}")
        train_pred_df = load_classified(train_output)
    else:
        train_pred_df = classify_texts(
            llm_client,
            train_df["text"].tolist(),
            filtered_criteria_descriptions,
            train_output,
            args.num_workers,
            domain_info,
        )

    train_pred_df = train_pred_df.sort_values(by="index")
    train_pred_df = train_pred_df.set_index('index').reset_index()

    if test_output.exists():
        logger.info(f"Loading existing classification from {test_output}")
        test_pred_df = load_classified(test_output)
    else:
        test_pred_df = classify_texts(
            llm_client,
            test_df["text"].tolist(),
            filtered_criteria_descriptions,
            test_output,
            args.num_workers,
            domain_info,
        )
    test_pred_df = test_pred_df.sort_values(by="index")
    test_pred_df = test_pred_df.set_index('index').reset_index()
    test_pred_df["label"] = test_df["label"].map(lambda x: trainer.class_to_index[x])

    # Attach numeric label indices used by Snorkel
    # train_pred_df["label"] = train_df["label"].map(lambda x: trainer.class_to_index[x])

    # Generate weak label matrices for all splits
    for name, df in {"train": train_pred_df, "test": test_pred_df, "dev": dev_pred_df}.items():
        L = trainer.applier.apply(df)
        wl_path = iter_dir / "weak_labels" / f"{name}.jsonl"
        pd.DataFrame(L, columns=[lf.name for lf in trainer.lfs]).to_json(
            wl_path,
            orient="records",
            lines=True,
        )
        logger.info(f"Saved weak labels for {name} to {wl_path}")

    train_pred_df_wl = pd.DataFrame(trainer.applier.apply(train_pred_df), columns=[lf.name for lf in trainer.lfs])
    dev_pred_df_wl = pd.DataFrame(trainer.applier.apply(dev_pred_df), columns=[lf.name for lf in trainer.lfs])
    test_pred_df_wl = pd.DataFrame(trainer.applier.apply(test_pred_df), columns=[lf.name for lf in trainer.lfs])

    wl_path = iter_dir / "weak_labels" / f"test_2.jsonl"
    test_pred_df_wl.to_json(wl_path,
            orient="records",
            lines=True,)

    mv_trainer = MajorityVoteTrainer(strategy='hard', handle_ties='prior')
    mv_trainer.fit(train_pred_df_wl.drop(["text", "label"], axis=1, errors="ignore"))

    # Evaluate the mv label model on the test set
    preds = mv_trainer.predict(test_pred_df_wl.drop(["text", "label"], axis=1, errors="ignore"))
    df_save = test_pred_df[["text", "label"]]
    df_save["pred"] = preds
    save_path = iter_dir / "classified" / "preds_mv_test.csv"
    df_save.to_csv(save_path)
    metrics = compute_metrics(test_pred_df["label"], preds)
    metrics_path = iter_dir / "metrics" / "metrics_mv_test.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Evaluate the mv label model on the dev set
    preds = mv_trainer.predict(dev_pred_df_wl.drop(["text", "label"], axis=1, errors="ignore"))
    df_save = dev_pred_df[["text", "label"]]
    df_save["pred"] = preds
    save_path = iter_dir / "classified" / "preds_mv_dev.csv"
    df_save.to_csv(save_path)
    metrics = compute_metrics(dev_pred_df["label"], preds)
    metrics_path = iter_dir / "metrics" / "metrics_mv_dev.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    #
    # ds_trainer = DawidSkeneTrainer(max_iter=5000)
    # ds_trainer.fit(train_pred_df_wl.drop(["text", "label"], axis=1, errors="ignore"))

    # Evaluate the ds label model on the test set
    # preds = ds_trainer.predict(test_pred_df_wl.drop(["text", "label"], axis=1, errors="ignore"))
    # metrics = compute_metrics(test_pred_df["label"], preds)
    # metrics_path = iter_dir / "metrics" / "metrics_ds_test.json"
    # with open(metrics_path, "w", encoding="utf-8") as f:
    #     json.dump(metrics, f, ensure_ascii=False, indent=2)
    # logger.info(f"Saved metrics to {metrics_path}")

    # Evaluate the ds label model on the dev set
    # preds = ds_trainer.predict(dev_pred_df_wl.drop(["text", "label"], axis=1, errors="ignore"))
    # metrics = compute_metrics(dev_pred_df["label"], preds)
    # metrics_path = iter_dir / "metrics" / "metrics_ds_dev.json"
    # with open(metrics_path, "w", encoding="utf-8") as f:
    #     json.dump(metrics, f, ensure_ascii=False, indent=2)
    # logger.info(f"Saved metrics to {metrics_path}")

    model_path = iter_dir / "models" / "label_model.pkl"
    if model_path.exists():
        logger.info(f"Loading label model from {model_path}")
        trainer.label_model.load(str(model_path))
    else:
        trainer.fit(train_pred_df)
        # trainer.label_model.save(model_path)
        logger.info(f"Saved label model to {model_path}")

    # Evaluate the snorkel label model on the test set
    preds = trainer.predict(test_pred_df)
    df_save = test_pred_df[["text", "label"]]
    df_save["pred"] = preds
    save_path = iter_dir / "classified" / "preds_snorkel_test.csv"
    df_save.to_csv(save_path)
    metrics = compute_metrics(test_pred_df["label"], preds)
    metrics_path = iter_dir / "metrics" / "metrics_snorkel_test.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Evaluate the snorkel label model on the dev set
    preds = trainer.predict(dev_pred_df)
    df_save = dev_pred_df[["text", "label"]]
    df_save["pred"] = preds
    save_path = iter_dir / "classified" / "preds_snorkel_dev.csv"
    df_save.to_csv(save_path)
    metrics = compute_metrics(dev_pred_df["label"], preds)
    metrics_path = iter_dir / "metrics" / "metrics_snorkel_dev.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Gather texts the model got wrong for next iteration
    wrong_df = dev_df[preds != dev_pred_df["label"]][["text", "label"]]
    wrong = [
        {"text": t, "label": str(l)}
        for t, l in zip(wrong_df["text"], wrong_df["label"])
    ]
    wrong_path = iter_dir / "classified" / "wrong.jsonl"
    with open(wrong_path, "w", encoding="utf-8") as f:
        json.dump(wrong, f, ensure_ascii=False, indent=2)
    correct_df = dev_df[preds == dev_pred_df["label"]][["text", "label"]]
    correct = [
        {"text": t, "label": str(l)}
        for t, l in zip(correct_df["text"], correct_df["label"])
    ]
    correct_path = iter_dir / "classified" / "correct.jsonl"
    with open(correct_path, "w", encoding="utf-8") as f:
        json.dump(correct, f, ensure_ascii=False, indent=2)


    # Fit and evaluate Catboost on train
    # preds_snorkel_train = trainer.predict(train_pred_df)
    # preds_mv_train = mv_trainer.predict(train_pred_df_wl.drop(["text", "label"], axis=1, errors="ignore"))
    # train_pool = Pool(train_df["text"], label=preds_snorkel_train, text_features=[0])
    # test_pool = Pool(test_df["text"], label=test_df["label"], text_features=[0])
    # model = CatBoostClassifier(loss_function="MultiClass", iterations=500, verbose=True)
    # model.fit(train_pool, verbose=True)
    # preds_cb_train = model.predict(test_pool)
    # metrics = compute_metrics(test_df["label"], preds_cb_train)
    # metrics_path = iter_dir / "metrics" / "metrics_test_cb_train.json"
    # with open(metrics_path, "w", encoding="utf-8") as f:
    #     json.dump(metrics, f, ensure_ascii=False, indent=2)
    # logger.info(f"Saved metrics to {metrics_path}")

    # Fit and evaluate Bert on train
    preds_snorkel_train = trainer.predict(train_pred_df)
    preds_mv_train = mv_trainer.predict(train_pred_df_wl.drop(["text", "label"], axis=1, errors="ignore"))
    model_bert, tokenizer_bert, label_encoder_bert, preds_bert = train_bert(train_df["text"], preds_mv_train, test_df["text"],
                                                        valid_texts=dev_df["text"], valid_labels=dev_df["label"],
                                                        model_name=config.MODEL_NAME_BERT,
                                                        epochs=config.EPOCHS_FIT_BERT, max_len=config.MAX_LEN_BERT)
    df_save = test_pred_df[["text", "label"]]
    df_save["pred"] = preds_bert
    save_path = iter_dir / "classified" / "preds_bert_test.csv"
    df_save.to_csv(save_path)
    metrics = compute_metrics(test_df["label"], preds_bert)
    metrics_path = iter_dir / "metrics" / "metrics_test_bert_train.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Finetune and evaluate Bert on dev
    # model_bert, tokenizer_bert, label_encoder_bert, preds_bert = train_bert(dev_df["text"], dev_df["label"], test_df["text"],
    #                                                     model=model_bert, tokenizer=tokenizer_bert, label_encoder=label_encoder_bert,
    #                                                     epochs=config.EPOCHS_CFT_BERT, max_len=config.MAX_LEN_BERT)
    # metrics = compute_metrics(test_pred_df["label"], preds_bert)
    # metrics_path = iter_dir / "metrics" / "metrics_test_bert_cft.json"
    # with open(metrics_path, "w", encoding="utf-8") as f:
    #     json.dump(metrics, f, ensure_ascii=False, indent=2)
    # logger.info(f"Saved metrics to {metrics_path}")

    return wrong, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--max_iter", type=int, default=1)
    parser.add_argument("--accuracy_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dev_size", type=int, default=1000)
    args = parser.parse_args()

    for _ in range(1):
        if args.dataset == "banking77":
            prepare_banking77_dataset(args.dev_size)
        if args.dataset == "chemprot":
            prepare_chemprot_dataset(args.dev_size)
        if args.dataset == "claude9":
            prepare_claude9_dataset(args.dev_size)
        if args.dataset == "tarif":
            prepare_tarif_dataset(args.dev_size)

        errors = None
        correct = None

        max_exp = 0
        for item in os.listdir(Path(args.output_dir) / args.dataset):
            if item.startswith("exp_"):
                item_num = int(item[4:])
                if item_num > max_exp:
                    max_exp = item_num
        exp_id = "exp_" + str(max_exp + 1)
        ensure_dir(Path(args.output_dir) / args.dataset / exp_id)

        json.dump({
            "dataset": args.dataset,
            "dev_size": args.dev_size,
            "accuracy_threshold": args.accuracy_threshold,
            "max_iter": args.max_iter,
            "llm_for_generation": config.LLM_FOR_GENERATION,
            "llm_for_classify": config.LLM_FOR_CLASSIFY,
            "classify_mode": config.CLASSIFY_MODE,
            "bert_name": config.MODEL_NAME_BERT
        }, open(Path(args.output_dir) / args.dataset / exp_id / "exp_config.json", "w"))

        for i in range(args.max_iter):
            errors, correct = run_iteration(args, i, exp_id, errors, correct)


if __name__ == "__main__":
    logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stderr, level="INFO") 
    main()
