"""Organ and technology vocabularies, ported from STPath's ``stpath/utils/constants.py``.

Upstream: https://github.com/Graph-and-Geometric-Learning/STPath

Only the two vocabularies the network actually consumes are kept. STPath's
``EncodeInputs`` embeds image features, gene tokens, technology and organ — the
species and cancer/domain-annotation vocabularies upstream also defines are
never read by the released checkpoint, so they are not ported.

Token order is load-bearing: the embedding rows in ``stfm.pth`` are indexed by
position in these lists. Do not reorder or insert.
"""

TECH_VOC = ["<pad>", "Spatial Transcriptomics", "Visium", "Xenium", "Visium HD"]

TECH_ALIGN = {
    "ST": "Spatial Transcriptomics",
    "Visium": "Visium",
    "Xenium": "Xenium",
    "VisiumHD": "Visium HD",
    "Visium HD": "Visium HD",
    "Spatial Transcriptomics": "Spatial Transcriptomics",
}

ORGAN_VOC = [
    "<pad>",
    "<mask>",
    "<unk>",
    "Spinal cord",
    "Brain",
    "Breast",
    "Bowel",
    "Skin",
    "Heart",
    "Kidney",
    "Prostate",
    "Lung",
    "Liver",
    "Uterus",
    "Bone",
    "Muscle",
    "Eye",
    "Pancreas",
    "Mouth",
    "Ovary",
    "Glioma",
    "Glioblastoma",
    "Stomach",
    "Colon",
    "Others",
]

ORGAN_ALIGN = {
    "Spinal cord": "Spinal cord",
    "Brain": "Brain",
    "Breast": "Breast",
    "Bowel": "Bowel",
    "Skin": "Skin",
    "Heart": "Heart",
    "Kidney": "Kidney",
    "Prostate": "Prostate",
    "Lung": "Lung",
    "Liver": "Liver",
    "Uterus": "Uterus",
    "Bone": "Bone",
    "Muscle": "Muscle",
    "Eye": "Eye",
    "Pancreas": "Pancreas",
    "breast": "Breast",
    "brain": "Brain",
    "kidney": "Kidney",
    "heart": "Heart",
    "skin": "Skin",
    "liver": "Liver",
    "pancreas": "Pancreas",
    "mouth": "Mouth",
    "ovary": "Ovary",
    "prostate": "Prostate",
    "glioma": "Glioma",
    "glioblastoma": "Glioblastoma",
    "stomach": "Stomach",
    "colon": "Colon",
    "lung": "Lung",
    "muscle": "Muscle",
    "Bladder": "Others",
    "Lymphoid": "Others",
    "Cervix": "Others",
    "Lymph node": "Others",
    "Ovary": "Others",
    "Embryo": "Others",
    "Lung/Brain": "Others",
    "Kidney/Brain": "Others",
    "Placenta": "Others",
    "Whole organism": "Others",
    "thymus": "Others",
    "joint": "Others",
    "undifferentiated pleomorphic sarcoma": "Others",
    "largeintestine": "Others",
    "lacrimal gland": "Others",
    "leiomyosarcoma": "Others",
    "endometrium": "Others",
    "brain+kidney": "Others",
    "cerebellum": "Others",
    "cervix": "Others",
    "colorectal": "Others",
    "lymphnode": "Others",
}

#: Sizes the released stfm.pth was trained with; used to validate the vocabularies
#: against the checkpoint's embedding tables.
N_TECH = len(TECH_VOC)
N_ORGANS = len(ORGAN_VOC)
