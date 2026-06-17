# diagnostics.py (project2)
# Ryan McGough 


from openai import OpenAI
from probability4e import *


T, F = True, False

# replace this with your api key
api_key = ''

client = OpenAI (
    api_key = api_key,
    base_url = "https://ellm.nrp-nautilus.io/v1"
)


# The prompt we're sending to OSS.
# This is tested for OSS from the LLM provider for CSUF. Running different LLMs *shouldn't* make a difference,
# but something to be mindful of regardless.
system_prompt = (
    "You are a probabilistic reasoning engine. "
    "Compute posterior probabilities using the Asia Bayesian network below.\n\n"
 
    "--- ASIA BAYESIAN NETWORK ---\n"
    "Prior probabilities:\n"
    "  P(Asia=T)    = 0.01\n"
    "  P(Smoking=T) = 0.50\n\n"

    "Conditional probabilities:\n"
    "  P(TB=T | Asia=T)        = 0.05,  P(TB=T | Asia=F)        = 0.01\n"
    "  P(Cancer=T | Smoking=T) = 0.10,  P(Cancer=T | Smoking=F) = 0.01\n"
    "  P(Bronchitis=T | Smoking=T) = 0.60, P(Bronchitis=T | Smoking=F) = 0.30\n\n"
    "  TBorC = True if TB=True OR Cancer=True  (deterministic OR)\n\n"
    "  P(Xray=Abnormal | TBorC=T) = 0.99, P(Xray=Abnormal | TBorC=F) = 0.05\n"
    "  P(Dyspnea=Present | TBorC=T, Bronchitis=T) = 0.90\n"
    "  P(Dyspnea=Present | TBorC=T, Bronchitis=F) = 0.70\n"
    "  P(Dyspnea=Present | TBorC=F, Bronchitis=T) = 0.80\n"
    "  P(Dyspnea=Present | TBorC=F, Bronchitis=F) = 0.10\n\n"
 
    "--- OUTPUT RULES (STRICT) ---\n"
    "- Do NOT show any working, explanation, math, or markdown.\n"
    "- Reply with ONLY 3 lines, nothing else:\n"
    "TB: <probability as decimal>\n"
    "Cancer: <probability as decimal>\n"
    "Bronchitis: <probability as decimal>\n\n"

    "Example of a valid reply:\n"
    "TB: 0.0014\n"
    "Cancer: 0.0142\n"
    "Bronchitis: 0.3142\n"
)


class Diagnostics:
    """ Use a Bayesian network to diagnose between three lung diseases """


    # Removed all the old code for the sake of cleanliness.
    def __init__(self):
        pass
    

    def diagnose (self, asia, smoking, xray, dyspnea):
        

        # Build the evidence string from the dropdown values
        evidence = "Patient evidence:\n"
        evidence += f"  Visit to Asia : {asia}\n"     # "Yes", "No", or "NA" (unknown)
        evidence += f"  Smoking       : {smoking}\n"  # "Yes", "No", or "NA"
        evidence += f"  Xray result   : {xray}\n"     # "Abnormal", "Normal", or "NA"
        evidence += f"  Dyspnea       : {dyspnea}\n"  # "Present", "Absent", or "NA"
        print(f"Values sent to the LLM:\n{evidence}")
 

        # Sending the Bayesian network prompt + evidence to the LLM
        response = client.chat.completions.create(
            model = "gpt-oss",
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": evidence}
            ]
        )
 

        # Extract the raw text reply
        llm_reply = response.choices[0].message.content
        print(f"LLM Reply:\n{llm_reply}")
 

        # Parse the three labeled lines: "TB: 0.xx", "Cancer: 0.xx", "Bronchitis: 0.xx"
        # If the LLM provides a reply not in the correct format, this won't work and will need to try again. 
        probabilities = {}
        for line in llm_reply.strip().splitlines():
            if ":" in line:
                label, value = line.split(":")
                probabilities[label.strip()] = float(value.strip())
 

        p_tb         = probabilities["TB"]
        p_cancer     = probabilities["Cancer"]
        p_bronchitis = probabilities["Bronchitis"]


        # Extracting the values from the reply
        print(f"P(TB)         = {p_tb:.4f}")
        print(f"P(Cancer)     = {p_cancer:.4f}")
        print(f"P(Bronchitis) = {p_bronchitis:.4f}")
 

        # Choosing the most likely disease
        candidates = [("TB", p_tb), ("Cancer", p_cancer), ("Bronchitis", p_bronchitis)]
        disease, probability = max(candidates, key=lambda x: x[1])
        print(f"Diagnosis: {disease} with probability {probability:.4f}")


        return [disease, probability]