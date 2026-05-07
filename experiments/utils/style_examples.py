"""
Style exemplar pools used to fit the casual / formal steering direction.

Each pool feeds :func:`utils.perplexity.concept_steering_vector`; the
mean-of-differences between FORMAL and CASUAL pools gives the direction
added to the residual stream during steered generation.
"""

FORMAL_EXAMPLES = [
    "I believe this proposal merits careful consideration.",
    "The analysis demonstrates significant improvements in performance.",
    "We respectfully submit our findings for your review.",
    "The organization has implemented comprehensive safety protocols.",
    "Our investigation reveals several important conclusions.",
    "The committee recommends immediate action on this matter.",
    "This methodology ensures reproducible and reliable results.",
    "We have thoroughly examined all available evidence.",
    "The data indicates a statistically significant correlation.",
    "Further research is warranted to confirm these findings.",
    "The results are consistent with the theoretical predictions.",
    "We acknowledge several limitations of the present study.",
    "The experimental protocol was designed to minimize bias.",
    "These findings have important implications for future work.",
    "The sample size was determined using power analysis.",
    "We conducted a systematic review of the literature.",
]

CASUAL_EXAMPLES = [
    "Yo, you gotta check this out, it's wild!",
    "Dude, these results are insane, way better than before!",
    "Honestly, just take a look - it speaks for itself.",
    "They basically fixed everything, super solid now.",
    "So yeah, turns out we were right about the whole thing.",
    "The team's like, we need to do something about this ASAP.",
    "This approach totally nails it every single time.",
    "We dug into all the data and here's the deal.",
    "No cap, this is the best we've seen so far.",
    "Can't believe how well this actually works tbh.",
    "OK so the numbers are kinda crazy good actually.",
    "Ngl this has some issues but it mostly slaps.",
    "We ran a bunch of tests and wow, just wow.",
    "The whole thing is way more interesting than expected.",
    "Literally all the data points in the same direction.",
    "We looked at everything and yeah, it checks out.",
]
