## 📊 Project structure for enterprise Supermarket Sales Prediction flows

- Build machine learning models using regressor algorithm
    - LinearRegression
    - XGBRegressor
    - RandomForestRegressor
- Build LLM end-to-end
    - Choose an LLM to fine-tune based on cost, resources, latency, security and accurate on answering to users
    - Host the LLMs or cloud server
    - Develop LLM Powered application to ensure deployed on production and proper testing
- Observability setup to monitor latency, logging Machine learning models and LLMs Fine-tuned
- Build Streamlit for User Interface Forecasting dashboard
- Alert using slack to notify on error logging in models, LLMs, & servers
- CI/CD setp to deploy for all evaluation benchmarks passed. To integrate recycle provided system deployment

## 📝 Notes
- In practice, strong RAG evaluation combines:
    - Retrieval checks → Did we fetch the right information?
    - Answer checks → Did we explain it correctly?
    - Continuous testing → Are we improving over time?