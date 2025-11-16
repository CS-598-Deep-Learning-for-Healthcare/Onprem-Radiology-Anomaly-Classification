import json

# ==============================
# CONFIG
# ==============================

V1_PATH = "multi_label_dataset_final_v1.json"
V2_PATH = "multi_label_dataset_final_v2.json"
V3_PATH = "multi_label_dataset_final_v3.json"
OUTPUT_PATH = "multi_label_dataset_final.json"


# ==============================
# LOAD DATA
# ==============================

def load_json_list(path):
    with open(path, "r") as f:
        return list(json.load(f))


file1 = load_json_list(V1_PATH)
file2 = load_json_list(V2_PATH)
file3 = load_json_list(V3_PATH)

if not (len(file1) == len(file2) == len(file3)):
    raise ValueError(
        f"Input files have different lengths: "
        f"{len(file1)}, {len(file2)}, {len(file3)}"
    )

n = len(file1)
print(f"Loaded {n} examples from each of v1/v2/v3.")


# ==============================
# MERGE LOGIC
# ==============================

final_dict = []
count_disagree = 0      # sentences where at least one label disagrees
count_all_diff = 0      # sentences where all three labels are different
count_majority_resolved = 0  # sentences resolved by majority+confidence

for i in range(n):
    r1 = file1[i]
    r2 = file2[i]
    r3 = file3[i]

    labels = [r1["Result"], r2["Result"], r3["Result"]]
    scores = [r1["confidence"], r2["confidence"], r3["confidence"]]

    sentence_dict = {}

    # ---------- CASE 1: All three runs agree ----------
    if labels[0] == labels[1] == labels[2]:
        sentence_dict["Result"] = labels[0]
        sentence_dict["Context"] = r1["Context"]
        sentence_dict["sentence_index"] = r1["sentence_index"]
        sentence_dict["confidence"] = sum(scores) / 3.0

        # support both old (text_file) and new (parquet) metadata
        if "text_file" in r1:
            sentence_dict["text_file"] = r1["text_file"]
        if "parquet_file" in r1:
            sentence_dict["parquet_file"] = r1["parquet_file"]
        if "row_index" in r1:
            sentence_dict["row_index"] = r1["row_index"]

    else:
        # Some disagreement between the three runs
        count_disagree += 1

        unique_labels = list(set(labels))

        # ---------- CASE 2: All three labels different ----------
        if len(unique_labels) == 3:
            # Completely inconsistent → drop this sentence
            count_all_diff += 1
            sentence_dict = {}

        # ---------- CASE 3: Exactly two labels present (e.g., Yes, Yes, No) ----------
        else:
            # For two unique labels, compute count and mean confidence per label
            label_a = unique_labels[0]
            label_b = unique_labels[1]

            count_a = 0
            count_b = 0
            sum_a = 0.0
            sum_b = 0.0

            for lbl, sc in zip(labels, scores):
                if lbl == label_a:
                    count_a += 1
                    sum_a += sc
                elif lbl == label_b:
                    count_b += 1
                    sum_b += sc

            mean_a = sum_a / count_a if count_a > 0 else 0.0
            mean_b = sum_b / count_b if count_b > 0 else 0.0

            # We only resolve if the majority label ALSO has higher avg confidence
            if count_a > count_b and mean_a > mean_b:
                chosen_label = label_a
                chosen_conf = mean_a
            elif count_b > count_a and mean_b > mean_a:
                chosen_label = label_b
                chosen_conf = mean_b
            else:
                # No clear winner → drop this sentence
                chosen_label = None

            if chosen_label is not None:
                count_majority_resolved += 1
                sentence_dict["Result"] = chosen_label
                sentence_dict["Context"] = r1["Context"]
                sentence_dict["sentence_index"] = r1["sentence_index"]
                sentence_dict["confidence"] = chosen_conf

                if "text_file" in r1:
                    sentence_dict["text_file"] = r1["text_file"]
                if "parquet_file" in r1:
                    sentence_dict["parquet_file"] = r1["parquet_file"]
                if "row_index" in r1:
                    sentence_dict["row_index"] = r1["row_index"]

    # If we built a consensus record, keep it
    if sentence_dict:
        final_dict.append(sentence_dict)


# ==============================
# STATS & SAVE
# ==============================

print("total length of sentence           :", n)
print("sentences with any disagreement    :", count_disagree)
print("sentences with three different lbls:", count_all_diff)
print("resolved by majority+confidence    :", count_majority_resolved)
print("unresolved (dropped)               :", count_disagree - count_all_diff - count_majority_resolved)
print("final usable sentences             :", len(final_dict))

with open(OUTPUT_PATH, "w") as out_file:
    json.dump(final_dict, out_file)

print(f"Saved merged dataset to {OUTPUT_PATH}")
