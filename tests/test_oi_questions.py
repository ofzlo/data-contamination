import json

from contam.data import oi_questions as oq

ROOT, ANIMAL, BIRD, DOG, MAMMAL, PERSON, MAN, LAMP, TREE = "/m/0bl9f", "/m/a", "/m/b", "/m/d", "/m/m", "/m/p", "/m/man", "/m/l", "/m/t"
EYE, STAIRS, ACC = "/m/eye", "/m/st", "/m/acc"
NAMES = {ANIMAL: "Animal", BIRD: "Bird", DOG: "Dog", MAMMAL: "Mammal", PERSON: "Person", MAN: "Man", LAMP: "Lamp", TREE: "Tree", "/m/n": "Nature",
         EYE: "Human eye", STAIRS: "Stairs", ACC: "Diatonic button accordion"}
NOT_BOXABLE = {"/m/n", ACC}
HIER = {"LabelName": ROOT, "Subcategory": [
    {"LabelName": ANIMAL, "Subcategory": [{"LabelName": BIRD}, {"LabelName": MAMMAL, "Subcategory": [{"LabelName": DOG}]}]},
    {"LabelName": PERSON, "Subcategory": [{"LabelName": MAN}]},
    {"LabelName": LAMP}, {"LabelName": TREE},
]}


def tag(text):
    return [(w.lower(), "NOUN") for w in text.replace(".", "").split()]


def test_hierarchy(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps(HIER))
    anc = oq.load_hierarchy(p)
    assert anc[DOG] == {ANIMAL, MAMMAL}
    assert anc[MAN] == {PERSON}
    assert ROOT not in anc and ANIMAL not in anc


def test_phrase():
    assert oq.phrase("Human eye")[1] == "Is there an eye in the image?" and "body_part" in oq.phrase("Human eye")[2]
    assert oq.phrase("Human body")[1] == "Is there a human body in the image?"
    assert oq.phrase("Footwear")[1] == "Is there any footwear in the image?"
    assert oq.phrase("Stairs")[1] == "Are there any stairs in the image?"
    assert oq.phrase("Ipod")[1] == "Is there an iPod in the image?"
    assert "parenthesis" in oq.phrase("Bat (Animal)")[2]
    assert "ends_in_s" in oq.phrase("Maracas-like things")[2]
    assert "capitalized" in oq.phrase("Dairy Thing")[2]
    assert oq.phrase("Lamp")[2] == []
    assert oq.phrase("Glasses")[1] == "Are there any eyeglasses in the image?"


def test_mentions_direct_descendant_synonym():
    anc = {DOG: {ANIMAL, MAMMAL}, BIRD: {ANIMAL}, MAN: {PERSON}}
    mm = oq.MentionMatcher(NAMES, anc)
    m = mm.mentioned_mids(tag("Two dogs and a guy"))
    assert m == {DOG, MAN}
    assert mm.mention_type(DOG, m) == "direct"
    assert mm.mention_type(MAMMAL, m) == "descendant"
    assert mm.mention_type(PERSON, m) == "descendant"
    assert mm.mention_type(LAMP, m) == "none"


def _build(labels, refs, train_idx=0, k=1, min_neg=1, b1=(), dep=None):
    ann = [{"image_id": 0, "open_images_id": "x", "file_name": "0.jpg", "domain": "near", "refs": refs}]
    split = {"name": "A", "seen": [0], "train_ref_idx": {"0": train_idx}}
    anc = {DOG: {ANIMAL, MAMMAL}, BIRD: {ANIMAL}, MAMMAL: {ANIMAL}, MAN: {PERSON}}
    boxable = set(NAMES) - NOT_BOXABLE
    return oq.build_questions(ann, split, {"x": labels}, NAMES, boxable, anc, {r: tag(r) for r in refs}, k_pos=k, k_neg=k, min_neg=min_neg, b1_overlap=b1, depictions=dep)


def L(c, src="verification"):
    return {"confidence": c, "source": src}


def test_build_filters_and_mentions():
    labels = {DOG: L(1), TREE: L(1), "/m/n": L(1), MAMMAL: L(0), LAMP: L(0), BIRD: L(0), MAN: L(0)}
    rows, stats = _build(labels, ["A dog under a tree", "A dog with a man"])
    by = {r["label_mid"]: r for r in rows}
    assert "/m/n" not in by and stats["drops"]["pos_not_boxable"] == 1          # scene label, not an object
    assert MAMMAL not in by and stats["drops"]["neg_ancestor_of_positive"] == 1  # contradicts Dog present
    assert MAN not in by and stats["drops"]["neg_mentioned_in_refs"] == 1        # a reference says "man"
    assert by[DOG]["mention_train"] == "direct" and by[DOG]["n_refs_mentioning"] == 2
    assert by[TREE]["mention_train"] == "direct" and by[TREE]["n_refs_mentioning"] == 1
    assert by[LAMP]["answer"] == "no" and by[LAMP]["mention_train"] == "none"
    assert sum(r["in_sample"] for r in rows) == 2
    assert {r["answer"] for r in rows if r["in_sample"]} == {"yes", "no"}


def test_build_train_ref_only_for_mention_train():
    rows, _ = _build({DOG: L(1), LAMP: L(0)}, ["A cute animal", "A dog on grass"], train_idx=0)
    dog = next(r for r in rows if r["label_mid"] == DOG)
    assert dog["mention_train"] == "none" and dog["n_refs_mentioning"] == 1


def test_sampling_deterministic_and_ineligible():
    labels = {DOG: L(1), TREE: L(1), LAMP: L(0), BIRD: L(0)}
    a, _ = _build(labels, ["x"], k=1)
    b, _ = _build(labels, ["x"], k=1)
    assert [r["in_sample"] for r in a] == [r["in_sample"] for r in b]
    c, stats = _build(labels, ["x"], k=3)
    assert stats["images"] == {"too_few_labels": 1} and not any(r["in_sample"] for r in c)


def test_negatives_boxable_and_variable_count():
    # 3 yes, 1 boxable no + 1 non-boxable no: eligible with k_pos=3, keeps the single no
    labels = {DOG: L(1), TREE: L(1), EYE: L(1), LAMP: L(0), ACC: L(0)}
    rows, stats = _build(labels, ["x"], k=3, min_neg=1)
    assert stats["images"] == {"eligible": 1} and stats["drops"]["neg_not_boxable"] == 1
    s = [r for r in rows if r["in_sample"]]
    assert sorted(r["answer"] for r in s) == ["no", "yes", "yes", "yes"]
    assert next(r for r in rows if r["label_mid"] == EYE)["is_body_part"]
    rows, stats = _build({DOG: L(1), TREE: L(1), EYE: L(1), ACC: L(0)}, ["x"], k=3, min_neg=1)
    assert stats["images"] == {"too_few_labels": 1}                                  # no boxable no left


def test_main_analysis_filter():
    labels = {DOG: L(1), MAMMAL: L(1), PERSON: L(1), LAMP: L(0)}
    rows, stats = _build(labels, ["x"], k=3)
    by = {r["label_mid"]: r for r in rows}
    assert by[MAMMAL]["in_sample"] and by[MAMMAL]["is_parent_class"] and not by[MAMMAL]["main_analysis"]
    assert by[PERSON]["is_parent_class"] and by[PERSON]["main_analysis"]             # Person exception
    assert by[DOG]["main_analysis"] and by[LAMP]["main_analysis"]
    assert stats["counts_main"] == {"images": 1, "questions": 3, "no": 1, "yes/none": 2}
    rows, _ = _build(labels, ["x"], k=3, b1=[0])
    assert not any(r["main_analysis"] for r in rows) and all(r["b1_overlap"] for r in rows)
    assert all(r["split_half"] == "seen_half" for r in rows) and "group" not in rows[0]


def test_count_table_groups_and_review_files():
    labels = {DOG: L(1), TREE: L(1), EYE: L(1), LAMP: L(0), STAIRS: L(0)}
    rows, _ = _build(labels, ["A dog"], k=3)
    t = oq.count_table(rows, "main_analysis", {"seen": {0}, "unseen": set()})
    assert t["seen"]["questions"] == 5 and t["seen"]["yes/direct"] == 1 and t["unseen"]["questions"] == 0
    rev = {d["label_name"]: d for d in oq.phrase_review_rows(rows)}
    assert set(rev) == {"Human eye", "Stairs"} and rev["Stairs"]["reasons"] == "plural_noun"
    ann = [{"image_id": 0, "refs": ["A dog"]}]
    chk = oq.mention_check_rows(rows, ann, n_per_cell=1)
    assert {c["cell"] for c in chk} == {"yes/direct", "yes/none", "no"} and all(c["train_caption"] == "A dog" for c in chk)
    assert "group" not in chk[0] and "split_half" not in chk[0]


def test_ambiguous_excluded_not_sampled(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
                 f"x,xclick,{LAMP},1,0,1,0,1,0,0,0,1,0\nx,xclick,{TREE},1,0,1,0,1,0,0,0,0,0\ny,xclick,{LAMP},1,0,1,0,1,0,0,0,1,0\n")
    dep = oq.load_depictions(p, ["x"])
    assert dep == {"x": {LAMP: (1, 1), TREE: (1, 0)}}
    # the lamp is a drawing, Man is a subtype, Mammal-absent contradicts the person: all excluded, never sampled
    labels = {LAMP: L(1), TREE: L(1), PERSON: L(1), STAIRS: L(1), MAN: L(1), MAMMAL: L(0), BIRD: L(0)}
    rows, stats = _build(labels, ["x"], k=3, dep=dep)
    by = {r["label_mid"]: r for r in rows}
    assert by[LAMP]["excluded"] == "depiction" and by[MAN]["excluded"] == "person_subtype" and by[MAMMAL]["excluded"] == "human_present"
    assert not any(by[m]["in_sample"] for m in (LAMP, MAN, MAMMAL))
    assert {m for m, r in by.items() if r["in_sample"]} == {TREE, PERSON, STAIRS, BIRD}
    assert by[TREE]["excluded"] is None and stats["drops"]["pos_depiction"] == 1 and stats["drops"]["neg_human_present"] == 1
    # without a person, Mammal absent is a normal candidate
    rows, _ = _build({STAIRS: L(1), TREE: L(1), LAMP: L(1), MAMMAL: L(0)}, ["x"], k=3)
    assert next(r for r in rows if r["label_mid"] == MAMMAL)["excluded"] is None


def test_share_table_groups():
    rows, _ = _build({DOG: L(1), TREE: L(1), STAIRS: L(1), LAMP: L(0)}, ["A dog"], k=3)
    t = oq.share_table(rows, {1000: {0}, 30: set()}, unseen=set())
    assert len(t) == 4 and t[0]["group"] == {"1000": "seen", "30": "unexposed"} and "file_name" in t[0] and "cell" in t[0]
    assert oq.share_table(rows, {1000: set()}, unseen={0})[0]["group"] == {"1000": "unseen"}
