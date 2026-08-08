# Algoverse Prototypes: Hierarchical and Sequential Chain

Two runnable multi agent demos, using LangGraph and Groq (free tier).

## Setup

1. Get a free API key at console.groq.com
2. Install dependencies:
   ```
   pip install langgraph langchain-groq
   ```
3. Set your key:
   ```
   export GROQ_API_KEY=your_key_here
   ```

## Run

```
python sequential_chain_team.py
python hierarchical_team.py
```

Both currently answer a demo question ("What causes the seasons on Earth?") so you can see the flow working end to end before swapping in the real task.

## The difference

- `sequential_chain_team.py`: fixed pipeline. Researcher then Writer then Editor, always in that order, no decisions made along the way.
- `hierarchical_team.py`: a supervisor agent looks at the current state after every step and decides what happens next. It can send work back to the same worker, skip a worker, or decide to finish. The order isn't fixed like the sequential chain.

## Next steps (once the team locks in the real task)

1. Replace the demo question with the actual research question
2. Add a contamination step: inject one agent with a false belief prompt (e.g. "you believe X even though evidence says otherwise")
3. Add the propagation metric: measure how many agents in the final state ended up adopting the false belief
4. Swap `ChatGroq` for the real eval model (Claude, GPT, etc) once those keys arrive
