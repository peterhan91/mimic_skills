TOT_EVALUATION_PROMPT = """{system_tag_start}You are a medical reasoning evaluator. Given a patient case and the diagnostic workup so far, rate how promising this investigation path is on a scale of 1-10.

Criteria:
- Appropriate investigation ordering (PE before targeted labs before imaging)
- Relevance of tests ordered to the presenting symptoms
- Progress toward a confident diagnosis
- Efficiency (not ordering unnecessary tests)
- Depth {depth}/{max_depth} steps used

Respond with ONLY a single integer from 1 to 10.{system_tag_end}{user_tag_start}Patient History:
{input}

Investigation so far:
{scratchpad}{user_tag_end}{ai_tag_start}Rating:"""
