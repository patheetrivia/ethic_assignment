When designing a custom LLM-based agent, the first question to ask is 'How can my model
outperform a publicly available out-of-the-box model?' Most of the prompts listed under 
Requirements 1f (ii, iii, iv, and vi) are handled perfectly well by ChatGPT, so I 
focused the bulk of my effort on building a model with special ability to answer the first 
prompt ('Given [list of criteria], what companies should I consider?'). Un-customized models 
struggle with that sort of prompt that implicitly invoke a formula; they have no native means to efficiently 
synthesize the criteria, so they regurgitate articles that discuss companies that excel at each individual criterion. 

My agent instead takes the following steps to handle that prompt: 

        - 1. Construct a dataframe that includes, for every company in the S&P 500, some financial snapshots
        along with ESG values, courtesy of Yahoo Finance (see data/sp500_companies.csv)
        - 2. Use an LLM to determine that the user does in fact want a list of companies that maximize some function
        - 2. Use another LLM to figure out what criteria are of concern / how to weight them
        - 3. Construct a linear formula based on those criteria (see tools/ranker.py)
        - 4. Apply the formula to the pre-constructed dataframe 
        - 5. Share the results with the user, and offer to save the results to a csv (see agent/exports
        for example csvs)

Outside of creating recommendation lists, my agent is little more than a direct interface with OpenAI's
GPT-5-mini model. To reduce latency, I directly pass basic financial details to my agent (again from Yahoo
Finance) as context when the user inquires into an individual ticker, but I perform no other customization or prompt engineering.

My agent has significant limitations. In particular, when constructing a list of recommended stocks, it guesses
somewhat aimlessly how much a user wants to weight different criteria. (How 
exactly how should it quantify the relative importance of beta and environmental risk if a user asks for
'low-volatility companies with some sustainability focus'?) With more time, I would have enabled user 
specification of the exact weights and formula, and generally made the formula construction less brittle. I also would 
have supported more potential user criteria; presently, each company is scored on a small set of pre-determined criteria, 
but the agent ideally should support generating + storing scores for whatever other criteria the user is interested in.
Finally, my agent is completely unequipped to construct non-linear formulae; if a user asks, for instance, for 
candidate sets of stocks whose price movements are negatively correlated, my model would be at a loss. Integrating 
sophisticated portfolio construction / balancing into a LLM-based chatbot is a fairly deep problem, and I 
think I just scratched the surface of ways to properly implement it. 

To scale this agent, I would wrap my agent in a light web service (FastAPI). I would
of course move away from storing results locally, and would consider building out a simple
RAG to help my agent pore through longer sources (like company sustainability reports). I would track latency,
error rates, and token usages, to better determine which exact models are best suited to which tasks (I suspect my agent is probably
currently overusing the somewhat heavyweight GPT-5-mini model). I would build out validation mechanisms and
redundancies to ensure that a source going down (like Yahoo Finance) doesn't break my agent. I would also
push heavier jobs into the background so that the chat remains responsive to the user. There are,
unfortunately, maybe 200 other scaling and productionizing fixes my agent desperately needs.






    

