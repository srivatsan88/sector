from sector.sector_main import run_sector
import json


# Example reference document and input text
# reference_sentence = """
# The goal of AI is to perform tasks that would typically require human intelligence such as visual perception speech recognition decisionmaking and language translation.
# """

# input_sentence = "It can recognize speech and perform tasks like humans."

input_text = "In this call, the customer contacted the service provider about intermittent internet connectivity issues. The agent verified the customer's information and walked the customer through restarting the modem and refreshing the signal from their system. The issue seemed resolved after the refresh, and the customer was advised to monitor the connection, with the option of scheduling a technician visit if the problem continued."

reference_doc = """
Customer:
Hi, I'm having trouble with my internet connection. It keeps dropping intermittently. Can you help me?

Agent:
I’m sorry to hear that. Can I have your account number or the phone number linked to your account, please?

Customer:
Sure, it’s 123456789.

Agent:
Thank you. Let me pull up your account. While I do that, have you tried restarting your modem?

Customer:
Yes, I’ve restarted it several times, but the problem keeps happening.

Agent:
I see. There are no outages in your area, but I can see that your modem has been having trouble maintaining a stable connection. I’ll go ahead and refresh your signal from our end. This should stabilize the connection. Could you please power off your modem for a few minutes and then turn it back on?

Customer:
Okay, I’ll do that right now.

Agent:
Great, I’ll stay on the line while you do that.

(After a brief pause)

Customer:
I’ve turned it back on.

Agent:
Thank you. I’ve refreshed the signal, and everything looks good from my end now. Please monitor your connection, and if the issue persists, we can schedule a technician to check it on-site.

Customer:
Alright, thanks for your help.

Agent:
You’re welcome! If you have any other issues, don’t hesitate to call us again. Have a great day!
"""

input_text1 = "In this call, the customer raised a concern about being overcharged on their recent bill. After confirming the account details, the agent quickly identified a duplicate charge and initiated a refund. The customer was informed that the refund would reflect within 3-5 business days and be applied to the next billing cycle. The customer expressed satisfaction with the resolution."

reference_doc1 = """
Customer:
Hi, I’m calling because I’ve been overcharged on my last bill. I was charged twice for the same service.

Agent:
I apologize for the inconvenience. Can I please have your account number so I can look into it?

Customer:
Sure, it’s 987654321.

Agent:
Thank you. Let me check your recent billing history… I can see that there was indeed a duplicate charge for your service. I’m sorry about that. I’ll initiate a refund for the extra charge right away.

Customer:
Great, thanks. How long will it take for the refund to go through?

Agent:
The refund will be processed within 3-5 business days. You’ll see the credit applied to your next billing cycle.

Customer:
Okay, that works. Thanks for sorting this out.

Agent:
You’re welcome, and again, I apologize for the inconvenience. Is there anything else I can help you with today?

Customer:
No, that’s all for now.

Agent:
Alright, have a great day!

"""

matched_sentences,final_score = run_sector(
    input_text1,
    reference_doc1,
    max_window_size=3,  # Combine consecutive sentences if needed
    use_semantic=True,  # Set to True for semantic matching or False for simple sliding window
    combine_threshold=0.996,  # Threshold for combining sentences
    top_n_individual=2,
    top_n_aggregated=2,
    debug=False, 
    search='sequential',
    input_clean_fn=None,
    context_clean_fn=None,
    embed_fn=None
)

response = {
    "matched_sentences": [
        {
            "input_sentence": match["input_sentence"],
            "matched_reference": match["matched_reference"],
            "sentence_similarity_score": match["scores"]["geometric_mean_top_n"]
        }
        for match in matched_sentences
    ],
    "scores": {
        "content_word_similarity": final_score["average_scores"]["content_word_similarity"],
        "ngram_fuzzy_match_score": final_score["average_scores"]["ngram_fuzzy_match_score"],
        "levenshtein_similarity": final_score["average_scores"]["levenshtein_similarity"],
        "pos_based_alignment_score": final_score["average_scores"]["pos_based_alignment_score"],
        "word_coverage": final_score["average_scores"]["word_coverage"],
        "key_word_coverage": final_score["average_scores"]["key_word_coverage"],
        "geometric_mean_top_n": final_score["average_scores"]["geometric_mean_top_n"],
        "sector_context_similarity": final_score["sector_context_similarity"]
        }
    }

print(json.dumps(response, indent=2))
