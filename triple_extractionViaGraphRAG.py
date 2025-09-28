import os
import re
import json
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def llm_extract_triples_gemini_enhanced(
    text: str,
    title: str,
    model="gemini-2.5-flash",
    verify_against_text=True
) -> List[Dict[str, str]]:
    """
    Enhanced extraction using Gemini API with anti-hallucination measures.
    """
    # Skip if text is too short
    if len(text.strip()) < 100:
        return []

    prompt = f"""
You are an expert biomedical knowledge extractor.
Your task:
- Extract ONLY knowledge graph triples directly stated in the text.
- Each triple must have this format:
  {{"subject": "...", "predicate": "...", "object": "..."}}
- If NO valid triples are present, return [] (an empty list).
- Do NOT add information not explicitly written in the text.
- Do NOT guess or hallucinate.
- Keep entities and relations concise (e.g., "mice", "exposed to", "microgravity").

CRITICAL: Only extract what is EXPLICITLY mentioned. If uncertain, omit the triple.

Publication Title: {title}

Text:
{text}

Output ONLY a JSON list of triples. Nothing else.
"""

    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)

        if not response or not response.text:
            return []

        content = response.text.strip()

        # Parse JSON safely
        triples = parse_json_safely(content)

        # Basic filtering
        clean_triples = []
        for triple in triples:
            if validate_triple_structure(triple):
                if verify_against_text and verify_triple_in_text(triple, text):
                    clean_triples.append(normalize_triple(triple))
                elif not verify_against_text:
                    clean_triples.append(normalize_triple(triple))

        clean_triples = remove_duplicate_triples(clean_triples)
        clean_triples = filter_low_quality_triples(clean_triples)

        return clean_triples

    except Exception as e:
        print(f"Gemini extraction error: {e}")
        return []


def parse_json_safely(content: str) -> List[Dict]:
    """Multiple strategies to extract valid JSON"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Strategy 1: Find JSON array with regex
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        # Strategy 2: Try to fix common JSON issues
        content = content.replace("'", '"')  # Fix single quotes
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays
        
        try:
            return json.loads(content)
        except:
            print(f"Failed to parse JSON: {content[:200]}...")
            return []

def validate_triple_structure(triple: Dict) -> bool:
    """Validate triple has proper structure and content"""
    required_fields = ["subject", "predicate", "object"]
    
    # Check required fields exist
    if not all(field in triple for field in required_fields):
        return False
    
    # Extract and clean values
    subj = str(triple["subject"]).strip()
    pred = str(triple["predicate"]).strip()
    obj = str(triple["object"]).strip()
    
    # Basic content validation
    if not all([subj, pred, obj]):  # No empty strings
        return False
    
    if subj.lower() == obj.lower():  # No self-references
        return False
    
    if any(len(x) < 2 for x in [subj, pred, obj]):  # Minimum length
        return False
    
    if any(len(x) > 100 for x in [subj, pred, obj]):  # Maximum length (prevent hallucinated long text)
        return False
    
    # Filter out generic/meaningless triples
    generic_predicates = ["is", "are", "has", "have", "contains", "includes"]
    if pred.lower() in generic_predicates and len(obj) < 5:
        return False
    
    # Filter out triples that are just paper metadata
    if any(word in subj.lower() for word in ["author", "study", "paper", "article", "research"]) and \
       any(word in pred.lower() for word in ["published", "written", "conducted"]):
        return False
    
    return True

def verify_triple_in_text(triple: Dict, text: str) -> bool:
    """
    Verify that triple components actually appear in the source text.
    This is the key anti-hallucination check.
    """
    text_lower = text.lower()
    subj_lower = triple["subject"].lower()
    obj_lower = triple["object"].lower()
    
    # Both subject and object must appear in text
    if subj_lower not in text_lower or obj_lower not in text_lower:
        return False
    
    # For biological terms, allow some flexibility in matching
    subj_variations = generate_term_variations(triple["subject"])
    obj_variations = generate_term_variations(triple["object"])
    
    subj_found = any(var in text_lower for var in subj_variations)
    obj_found = any(var in text_lower for var in obj_variations)
    
    return subj_found and obj_found

def generate_term_variations(term: str) -> List[str]:
    """Generate variations of a term for flexible matching"""
    term_lower = term.lower()
    variations = [term_lower]
    
    # Add plural/singular variations
    if term_lower.endswith('s') and len(term_lower) > 3:
        variations.append(term_lower[:-1])  # Remove 's'
    else:
        variations.append(term_lower + 's')  # Add 's'
    
    # Add common biological abbreviations
    bio_abbrevs = {
        "deoxyribonucleic acid": "dna",
        "ribonucleic acid": "rna",
        "bone mineral density": "bmd",
        "cardiovascular": "cv",
        "central nervous system": "cns"
    }
    
    for full_term, abbrev in bio_abbrevs.items():
        if full_term in term_lower:
            variations.append(term_lower.replace(full_term, abbrev))
        if abbrev == term_lower:
            variations.append(full_term)
    
    return variations

def normalize_triple(triple: Dict) -> Dict[str, str]:
    """Normalize triple format for consistency"""
    return {
        "subject": triple["subject"].strip().lower(),
        "predicate": normalize_predicate(triple["predicate"].strip().lower()),
        "object": triple["object"].strip().lower()
    }

def normalize_predicate(predicate: str) -> str:
    """Normalize predicates to standard forms"""
    predicate = predicate.lower().strip()
    
    # Normalize to standard forms
    predicate_mapping = {
        "leads to": "causes",
        "results in": "causes", 
        "induces": "causes",
        "produces": "causes",
        "brings about": "causes",
        
        "reduces": "decreases",
        "lowers": "decreases",
        "diminishes": "decreases",
        
        "increases": "increases",
        "enhances": "increases",
        "boosts": "increases",
        "elevates": "increases",
        
        "prevents": "prevents",
        "inhibits": "prevents",
        "blocks": "prevents",
        "stops": "prevents",
        
        "affects": "influences",
        "impacts": "influences",
        "modifies": "influences"
    }
    
    return predicate_mapping.get(predicate, predicate)

def remove_duplicate_triples(triples: List[Dict]) -> List[Dict]:
    """Remove duplicate triples"""
    seen = set()
    unique_triples = []
    
    for triple in triples:
        # Create a normalized key for comparison
        key = (triple["subject"], triple["predicate"], triple["object"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)
    
    return unique_triples

def filter_low_quality_triples(triples: List[Dict]) -> List[Dict]:
    """Filter out low-quality triples that might be hallucinated"""
    quality_triples = []
    
    for triple in triples:
        subj, pred, obj = triple["subject"], triple["predicate"], triple["object"]
        
        # Skip overly generic triples
        generic_subjects = ["it", "this", "that", "they", "we", "study", "research"]
        if subj in generic_subjects:
            continue
        
        # Skip triples with numbers only (often measurement errors)
        if subj.isdigit() or obj.isdigit():
            continue
        
        # Skip triples with special characters (often parsing errors)
        if any(char in subj + pred + obj for char in "[]{}()@#$%^&*"):
            continue
        
        # Keep high-quality triples
        quality_triples.append(triple)
    
    return quality_triples

def validate_extraction_batch(triples: List[Dict], source_texts: List[str]) -> Dict[str, any]:
    """
    Validate a batch of extractions for quality assessment
    """
    if not triples:
        return {"valid": True, "quality_score": 0, "issues": ["no_triples_extracted"]}
    
    total_triples = len(triples)
    issues = []
    
    # Check for repeated subjects/objects (sign of hallucination)
    subjects = [t["subject"] for t in triples]
    objects = [t["object"] for t in triples]
    
    subject_repeats = len(subjects) - len(set(subjects))
    object_repeats = len(objects) - len(set(objects))
    
    if subject_repeats > total_triples * 0.5:
        issues.append("too_many_repeated_subjects")
    
    if object_repeats > total_triples * 0.5:
        issues.append("too_many_repeated_objects")
    
    # Check predicate diversity
    predicates = [t["predicate"] for t in triples]
    unique_predicates = len(set(predicates))
    
    if unique_predicates < 2 and total_triples > 5:
        issues.append("low_predicate_diversity")
    
    # Quality score (0-1)
    quality_score = 1.0
    quality_score -= (subject_repeats / total_triples) * 0.3
    quality_score -= (object_repeats / total_triples) * 0.3
    quality_score -= len(issues) * 0.1
    quality_score = max(0, quality_score)
    
    return {
        "valid": quality_score > 0.5,
        "quality_score": quality_score,
        "issues": issues,
        "stats": {
            "total_triples": total_triples,
            "unique_subjects": len(set(subjects)),
            "unique_predicates": unique_predicates,
            "unique_objects": len(set(objects))
        }
    }

# Example usage with quality monitoring
def extract_with_quality_control(text_chunks: List[Dict], client=None) -> Dict:
    """
    Extract triples with built-in quality control for hackathon demo
    """
    all_triples = []
    extraction_stats = {
        "total_chunks": len(text_chunks),
        "successful_extractions": 0,
        "failed_extractions": 0,
        "quality_issues": []
    }
    
    for chunk in text_chunks:
        text = chunk.get("text", "")
        title = chunk.get("metadata", {}).get("title", "Unknown")
        
        # Extract triples
        triples = llm_extract_triples_gemini_enhanced(
            text=text,
            title=title, 
            verify_against_text=True  # Enable strict verification
        )
        
        if triples:
            # Validate quality
            validation = validate_extraction_batch(triples, [text])
            
            if validation["valid"]:
                all_triples.extend(triples)
                extraction_stats["successful_extractions"] += 1
            else:
                extraction_stats["failed_extractions"] += 1
                extraction_stats["quality_issues"].extend(validation["issues"])
        else:
            extraction_stats["failed_extractions"] += 1
    
    return {
        "triples": all_triples,
        "stats": extraction_stats,
        "quality_summary": {
            "total_triples": len(all_triples),
            "success_rate": extraction_stats["successful_extractions"] / len(text_chunks) if text_chunks else 0,
            "avg_triples_per_chunk": len(all_triples) / max(1, extraction_stats["successful_extractions"])
        }
    }

# For hackathon demo
if __name__ == "__main__":
    # Test with sample space biology text
    sample_text = """
    Astronauts exposed to microgravity for extended periods experience significant bone loss. 
    Studies show that bone mineral density decreases by approximately 1-2% per month during spaceflight. 
    Exercise countermeasures can partially prevent this bone loss. Resistance training helps maintain muscle mass.
    """
    
    triples = llm_extract_triples_gemini_enhanced(
        text=sample_text,
        title="Bone Loss in Microgravity",
        verify_against_text=True
    )
    
    print("🔍 Extracted triples (anti-hallucination enabled):")
    for triple in triples:
        print(f"  {triple['subject']} --{triple['predicate']}--> {triple['object']}")
    
    print(f"\n✅ Extracted {len(triples)} verified triples")