# Product Backlog: Ecommerce AI

This backlog tracks the evolution of the AI-enabled recommendation engine.

| ID | Title | Priority | Description | Estimate | Status |
|:---|:---|:---|:---|:---|:---|
| **TECH-01** | UI Refinement: Recs Page | High | Replace the string return in `/recommendations` with a modern Glassmorphism template consistent with the Index page. | 3 pts | To Do |
| **ALGO-01** | Model Serialization | High | Ensure `svd_model.pkl` is versioned or script-regenerated to prevent loading errors. | 1 pt | Done |
| **FEAT-01** | Item Metadata Display | Medium | Enhance recommendations to include item Category and Price instead of just IDs. | 5 pts | To Do |
| **FEAT-02** | User Search Autocomplete | Medium | Implement suggested User IDs in the dashboard search bar based on available data. | 3 pts | To Do |
| **DEVOPS-01** | Automated Retraining | Low | Create a cron job or background task to retrain the SVD model weekly as new data arrives. | 8 pts | Pending |
| **SEC-01** | User Authentication | Medium | Implement a simple login system to protect the Intelligence Dashboard. | 5 pts | To Do |
