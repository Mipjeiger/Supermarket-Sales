## 📊 Project structure for enterprise Supermarket Sales Prediction flows

- Build Supermarket database in postgresql
    ![alt text](Database/images/F9178668-ED61-42E3-9669-123D99585FEF.png)
- Build Machine Learning models to help LLM judge prediction for reason
    - Build machine learning models using regressor algorithm for sales prediction
        ![alt text](Database/images/253B1E3B-D23A-462D-A8D1-C34C4B6F0E34_4_5005_c.jpeg)
    - Build entity Fraud prediction using ML classifier models and analysis factor to integrate with streamlit
        ![alt text](Database/images/1684AAFF-F18A-4BF6-AF43-3CE0E652AB37_4_5005_c.jpeg)
    - Build entity Abuse detection, analysis factor to integrate with streamlit, and Entity Security agent to fast investigate on LLM analysis in main app for production using ML models and LLMs
    ![alt text](Database/images/EC25BF46-F8B7-451E-AD04-C413C7BB6825.png)
- Build LLM end-to-end
    - Choose an LLM to fine-tune based on cost, resources, latency, security and accurate on answering to users
    - Host the LLMs or cloud server
    - Develop LLM Powered application to ensure deployed on production and proper testing
- Build Docker to create ML system components of tools end-to-end
    ![alt text](Database/images/CFA2A243-8DA6-47A8-9EB1-04F2BDCEC177_4_5005_c.jpeg.png)
- Build ML Pipeline to send ML models to MLflow for tracking & experimental
- Observability setup to monitor latency, logging Machine learning models and LLMs Fine-tuned
- Build Streamlit for User Interface Forecasting dashboard
    📈 Sales Prediction
    ![alt text](Database/images/845BE1AB-11AA-4ED8-B322-53EE7AC865D7_1_105_c.jpeg)
    🚨 Abuse Detection
    ![alt text](Database/images/063C7808-808F-47A7-93E0-359D5E832E2F_1_105_c.jpeg)
    ![alt text](Database/images/2DFC9CEF-16ED-47F6-8A49-D6D56E839D84_1_105_c.jpeg)
    🔐 Security Agent
    ![alt text](Database/images/4A434B34-9EE5-421C-9DC5-6BAFEC921DD8_1_105_c.jpeg)
    📊 Future Revenue Forecasting
    ![alt text](Database/images/460CA2A6-72F1-4E8B-A954-B260D34211CC.png)
- Build DVC (Data Version Control) to allow track and version large datasets, ML models, training pipelines, and git for data (Optional)
- Alert using slack to notify on error logging in models, LLMs, & servers
- CI/CD setp to deploy for all evaluation benchmarks passed. To integrate recycle provided system deployment
- Streamlit Deployment

## 📝 Notes
- In practice, strong RAG evaluation combines:
    - Retrieval checks → Did we fetch the right information?
    - Answer checks → Did we explain it correctly?
    - Continuous testing → Are we improving over time?

## ⚙️ Service tools
1. Slack
2. Streamlit
3. Docker
4. FastAPI
5. MLFlow
6. Grafana
7. Prometheus
8. PostgreSQL
9. Github actions
10. Kubernetes