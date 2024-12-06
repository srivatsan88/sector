import json
from pkg_resources import resource_filename

_properties_cache = None  # Module-level cache


def load_properties():
    # Load the properties.json file
    global _properties_cache
    if _properties_cache is None:  # Load properties if not already loaded
        properties_path = resource_filename("sector.data", "properties.json")
        with open(properties_path, "r") as f:
            _properties_cache = json.load(f)
    
    # Extract each section into separate variables
    affirmative_chain = {phrase for group in _properties_cache.get("affirmative_chain", []) for phrase in group}
    negative_chain = {phrase for group in _properties_cache.get("negative_chain", []) for phrase in group}
    bias_check = {phrase for group in _properties_cache.get("bias_check", []) for phrase in group}
    toxic_check = {phrase for group in _properties_cache.get("toxic_check", []) for phrase in group}
    hate_check = {phrase for group in _properties_cache.get("hate_check", []) for phrase in group}
    profanity_check = {phrase for group in _properties_cache.get("profanity_check", []) for phrase in group}

    # Load thresholds and other properties
    thresholds = _properties_cache.get("thresholds", {})
    other_properties = _properties_cache.get("other_properties", {})

    return {
        "affirmative_chain": affirmative_chain,
        "negative_chain": negative_chain,
        "bias_check": bias_check,
        "toxic_check": toxic_check,
        "hate_check": hate_check,
        "profanity_check": profanity_check,
        "thresholds": thresholds,
        "other_properties": other_properties
    }
