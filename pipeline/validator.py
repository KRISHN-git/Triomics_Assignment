import logging
logger = logging.getLogger(__name__)

VALID_TAXONOMY = {
    "cancer": ["primary_malignancy","metastasis","pre_malignant","benign","cancer_of_unknown_primary"],
    "cardiovascular": ["coronary","hypertensive","rhythm","vascular","structural","inflammatory_vascular"],
    "infectious": ["bacterial","viral","fungal","parasitic","spirochetal"],
    "metabolic_endocrine": ["diabetes","thyroid","genetic_metabolic","nutritional_deficiency","lipid","adrenal","pituitary"],
    "neurological": ["cerebrovascular","traumatic","seizure","functional","degenerative","neuromuscular"],
    "pulmonary": ["obstructive","acute_respiratory","structural","occupational","cystic"],
    "gastrointestinal": ["hepatic","biliary","upper_gi","lower_gi","inflammatory_bowel","functional_gi"],
    "renal": ["renal_failure","structural","glomerular","renovascular"],
    "hematological": ["cytopenia","coagulation","hemoglobinopathy"],
    "immunological": ["immunodeficiency","allergic","autoimmune","autoinflammatory","complement_deficiency"],
    "musculoskeletal": ["fracture","degenerative","crystal_arthropathy","connective_tissue_disorder"],
    "toxicological": ["poisoning","environmental_exposure"],
    "dental_oral": ["dental","temporomandibular"]
}

VALID_STATUSES = {"active", "resolved", "suspected"}


def validate_conditions(conditions: list, patient_id: str) -> list:
   
    valid = []
    for c in conditions:
        name        = c.get("condition_name", "UNNAMED")
        category    = c.get("category", "")
        subcategory = c.get("subcategory", "")
        status      = c.get("status", "")
        evidence    = c.get("evidence", [])

        if category not in VALID_TAXONOMY:
            logger.warning(f"[{patient_id}] DROPPED '{name}': invalid category '{category}'")
            continue

        if subcategory not in VALID_TAXONOMY[category]:
            logger.warning(f"[{patient_id}] DROPPED '{name}': invalid subcategory '{subcategory}'")
            continue

        if status not in VALID_STATUSES:
            logger.warning(f"[{patient_id}] FIXED status '{status}' → 'active' for '{name}'")
            c["status"] = "active"

        if not evidence:
            logger.warning(f"[{patient_id}] DROPPED '{name}': no evidence")
            continue

        valid.append(c)

    dropped = len(conditions) - len(valid)
    if dropped > 0:
        logger.info(f"[{patient_id}] Validator dropped {dropped} invalid conditions")

    return valid