import transformers as tr
import torch
import torch.nn.functional as F

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path, trust_remote_code=True)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
    scores,
    results,
    kFactor = 4,
) {
    for (const result of results) {
        const { first, second, outcome } = result;
        const firstScore = scores[first] ?? 1000;
        const secondScore = scores[second] ?? 1000;

        const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
        const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
        let sa = 0.5;
        if (outcome === 1) {
            sa = 1;
        } else if (outcome === -1) {
            sa = 0;
        }
        scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
        scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
    }
    return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

def contrastive_generation(amateur, expert, prompt, max_tokens=60) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amateur = amateur.to(device).eval()
    expert = expert.to(device).eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    # Keep track of sentence completion
    last_punctuation_pos = -1
    
    for i in range(max_tokens):
        with torch.no_grad():
            # Get logits
            amateur_logits = amateur(input_ids).logits[:, -1, :]
            expert_logits = expert(input_ids).logits[:, -1, :]
            
            # Stable log difference
            expert_lprobs = F.log_softmax(expert_logits, dim=-1)
            amateur_lprobs = F.log_softmax(amateur_logits, dim=-1)
            
            # Controlled contrast
            logits = expert_lprobs - 0.3 * amateur_lprobs  # Reduced amateur penalty
            
            # Numerically stable softmax
            probs = torch.softmax(logits / 0.6, dim=-1)
            
            # Penalize non-ending tokens in later steps
            if i > 30:  # After 30 tokens, encourage completion
                probs[:, tokenizer.eos_token_id] *= 2
                probs[:, tokenizer.convert_tokens_to_ids(".")] *= 1.5
                
            next_token = torch.multinomial(probs, num_samples=1)

        # Track punctuation for clean truncation
        if next_token.item() in [tokenizer.eos_token_id, 
                               tokenizer.convert_tokens_to_ids("."),
                               tokenizer.convert_tokens_to_ids(";")]:
            last_punctuation_pos = input_ids.shape[-1]
            
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Early exit if natural ending found
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Trim to last punctuation if needed
    if last_punctuation_pos != -1:
        input_ids = input_ids[:, :last_punctuation_pos+1]
        
    return tokenizer.decode(input_ids[0][inputs.input_ids.shape[1]:], 
                          skip_special_tokens=True).strip()


if __name__ == "__main__":
    amateur = tr.AutoModelForCausalLM.from_pretrained(amateur_path, torch_dtype=torch.bfloat16)
    expert = tr.AutoModelForCausalLM.from_pretrained(expert_path, torch_dtype=torch.bfloat16)
    
    result = contrastive_generation(amateur, expert, prompt)
    print(result)