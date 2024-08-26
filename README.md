
# Privacy-Preserving Multi-Disease Diagnosis using Federated Multi-Task Learning (FMTL)



This repository contains the research and code design developed as part of an IoT research project proposal for the EE5003 module at Dublin City University (DCU) for the academic year 2023-2024. The project is a proof of concept that explores multi-task classifier architectures within a federated learning environment.


## Project Overview

- In the context of the Internet of Medical Things (IoMT), maintaining patient data privacy and integrity in both outpatient and inpatient settings is crucial, especially when automating or enhancing medical diagnostics. Healthcare providers are often reluctant to share Electronic Health Record (EHR) data due to privacy concerns, which limits the potential for collaborative research and improved diagnostics across institutions.

- **Federated Learning (FL)** addresses these privacy concerns by enabling the training of classification and regression models across distributed devices without the need to share sensitive data. The security of FL can be further enhanced through techniques such as Differential Privacy, as well as robust encryption schemes like AES, Homomorphic Encryption, and Asymmetric Key Encryption.

- Certain diseases often co-occur, either causally or non-causally, as comorbidities or multimorbidities. Chronic diseases, in particular, share many diagnostic criteria and often require similar tests and examinations. With the integration of IoT sensors and smart devices into medical diagnostics, combining sensor data with EHR data offers a holistic approach to patient care, potentially uncovering new insights.

- This project proposes secure and privacy-preserving solutions to connect knowledge from multiple remote facilities. By leveraging federated multi-task learning (FMTL), the project aims to improve the generalization of disease detection models trained across multiple clients, each focusing on related but distinct illnesses.


- **Security Enhancements:**
  - The FL security can be augmented by incorporating Differential Privacy transmission channels along with robust security schemes such as AES, Homomorphic Encryption, and Asymmetric Key Encryption.

- **Medical Context:**
  - The project considers that certain groups of diseases can co-exist (co-morbidity and multi-morbidity) and that some chronic diseases share overlapping diagnostic criteria, which justifies the multi-task learning approach.
  - Integration of IoT sensors and smart devices in recent years provides valuable data when combined with a patient's Electronic Health Record (EHR), offering a holistic approach to patient diagnosis.

- **Performance Improvement Strategies:**
  - Additional approaches like stacking the multi-task model with other classification models can improve performance by adding features unique to each disease diagnosis set.
  - This method aligns with real medical scenarios where examinations are multi-step and progressively involve additional tests.


## Project Components

### 1. Disease Groups Explored

#### a. Long-term Hypertension and Coronary Heart Disease (CHD)

- This demo uses a simplified version of the Framingham Heart Study dataset, which is publicly available [here](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset). This dataset is limited, with an imbalanced class distribution and a small number of features and entries.

#### b. Diabetes Mellitus and Chronic Kidney Disease (CKD)

- A more complex dataset was created from scratch using the MIMIC-IV database, accessible via [PhysioNet](https://physionet.org/content/mimiciv/2.2/). This example aligns more closely with realistic IoT-EHR data recording, featuring more examples and realistic predictive scores, designed to work with heterogeneous and uneven data distributions across different healthcare providers.

### 2. Multi-Task Learning Model

- The project demonstrates two primary examples:

- **Framingham Demo:** A basic demonstration using a small, imbalanced dataset.
- **MIMIC-IV Final Application:** A more refined and realistic example using actual patient IoT and EHR data with realistic performance metrics.

### 3. Model Enhancement with Ensemble Stacking

- In addition to the multi-task model, the project explores further improvements through ensemble stacking. This approach involves stacking the FMTL model with an additional classifier model, such as scikit-learn's SVC module; a Support Vector Machine (SVM) variant, to enhance predictive performance.

## How to Run

### 1. Install Dependencies

- Install the required dependencies from the `requirements.txt` file. Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

### 2. Framingham Demo Example

- **Data Exploration and Preprocessing:**
  - Review the `framingham.csv` dataset and the `framingham.ipynb` notebook for data cleaning, processing, and initial testing.
  - The notebook defines expected metrics for the two disease labels:
    - `prevalentHyp`
    - `TenYearCHD`
  - Models used for evaluation include:
    - `TabNetMultiTask` classifier for multi-task reference.
    - Decision-tree-based single-task classifier `XGBoost` to assess each task independently.

- **Running Federated Learning Framework:**
  - The primary federated learning framework used is `flwr`.
  - Execute the shell/batch scripts located in the `framingham_demo/fl/` folder.
  - You should have three prompts running, displaying the following results:

    - **Server Output:**
      ```
      INFO :      Flower ECE: gRPC server running (30 rounds), SSL is disabled
      INFO :      [INIT]
      INFO :      Requesting initial parameters from one random client
      WARNING :   Failed to receive initial parameters from the client. Empty initial parameters will be used.
      INFO :      Evaluating initial global parameters
      INFO :
      INFO :      [ROUND 1]
      INFO :      configure_fit: strategy sampled 2 clients (out of 2)
      INFO :      aggregate_fit: received 2 results and 0 failures
      Initiating Stats
      Initiating Stats
      ({'Loss Task 1: ': 0.576942652463913, 'Loss Task 2: ': 0.704729026556015}, {'accuracy Task 1:': 0.7112561174551386, 'accuracy Task 2:': 0.43882544861337686})
      INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
      INFO :      aggregate_evaluate: received 2 results and 0 failures
      Checking Conditions
      INFO :
      INFO :      [ROUND 2]
      INFO :      configure_fit: strategy sampled 2 clients (out of 2)
      INFO :      aggregate_fit: received 2 results and 0 failures
      Initiating Stats
      Initiating Stats
      ({'Loss Task 1: ': 0.4935342982411385, 'Loss Task 2: ': 0.7701077789068222}, {'accuracy Task 1:': 0.8564437194127243, 'accuracy Task 2:': 0.3866231647634584})
      INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
      INFO :      aggregate_evaluate: received 2 results and 0 failures
      Checking Conditions
      ```

    - **Client #1 (Hypertension):**
      ```
      INFO :      Received: train message 3ed48c69-a960-4c9c-a816-6120d7d46d3e
      Epoch [1/5], Loss: 0.7513
      Epoch [2/5], Loss: 0.6751
      Epoch [3/5], Loss: 0.6259
      Epoch [4/5], Loss: 0.5178
      Epoch [5/5], Loss: 0.5233
      INFO :      Sent reply
      INFO :
      INFO :      Received: evaluate message b749b96f-ef87-441d-b98a-8cc34f8927c6
      Starting evaluation...
      Initiating Stats
      Loss: 0.5121812845269839
      Accuracy: 0.8
      INFO :      Sent reply
      INFO :
      INFO :      Received: train message 5f591feb-d523-4271-864a-c7f11c199be2
      Epoch [1/5], Loss: 0.4534
      Epoch [2/5], Loss: 0.4495
      Epoch [3/5], Loss: 0.4825
      Epoch [4/5], Loss: 0.3508
      Epoch [5/5], Loss: 0.4158
      INFO :      Sent reply
      INFO :
      INFO :      Received: evaluate message a2ebf496-d6b0-4a01-b757-7dce760bef11
      Starting evaluation...
      Initiating Stats
      Loss: 0.3864894186456998
      Accuracy: 0.8388888888888889
      INFO :      Sent reply
      ```

    - **Client #2 (CHD):**
      ```
      INFO :      Received: get_parameters message d8321452-2fa9-4af7-a7b0-1ddb2e4cce29
      INFO :      Sent reply
      INFO :
      INFO :      Received: train message cfa7e632-3533-41f1-90ae-bec12cc36141
      Epoch [1/5], Loss: 0.5979
      Epoch [2/5], Loss: 0.4791
      Epoch [3/5], Loss: 0.6008
      Epoch [4/5], Loss: 0.4432
      Epoch [5/5], Loss: 0.4759
      INFO :      Sent reply
      INFO :
      INFO :      Received: evaluate message a8b95fb3-8039-421f-b5dd-008fc1defaee
      Starting evaluation...
      Initiating Stats
      Loss: 0.4965183213353157
      Accuracy: 0.825
      INFO :      Sent reply
      INFO :
      INFO :      Received: train message 08b1a071-fc3a-4a70-9fa8-fbc93979cd70
      Epoch [1/5], Loss: 0.3324
      Epoch [2/5], Loss: 0.3534
      Epoch [3/5], Loss: 0.3884
      Epoch [4/5], Loss: 0.3560
      Epoch [5/5], Loss: 0.3841
      INFO :      Sent reply
      INFO :
      INFO :      Received: evaluate message 68879571-443f-43f5-8ff0-fef5145c429d
      Starting evaluation...
      Initiating Stats
      Loss: 0.44640904913345975
      Accuracy: 0.825
      INFO :      Sent reply
      ```

### 3. MIMIC-IV Final Application

- **Setup:**
  - Navigate to the `/main_mimic` folder and follow similar steps as in the Framingham demo.
  - *Optional:* To explore the scripts used for generating patients, features, and labels, refer to the `/sql` directory. Access to the MIMIC-IV database is required, which can be requested from [PhysioNet](https://physionet.org/content/mimiciv/2.2/).
  - The resulting data from these SQL queries are stored in the `/data` folder.

- **Running with Docker Compose:**
  - Ensure you have Docker installed:
    ```bash
    sudo apt-get install docker-ce docker-compose
    ```
  - In the `/fl` directory, run the following command:
    ```bash
    docker-compose up
    ```
  - The final output should resemble:
    ```
    Starting dockersupport_server_1 ... done
    Starting dockersupport_client1_1 ... done
    Starting dockersupport_client2_1 ... done
    Attaching to dockersupport_server_1, dockersupport_client2_1, dockersupport_client1_1
    server_1   | INFO :      Starting Flower server, config: num_rounds=30, no round_timeout
    server_1   | INFO :      Flower ECE: gRPC server running (30 rounds), SSL is disabled
    server_1   | INFO :      [INIT]
    server_1   | INFO :      Requesting initial parameters from one random client
    client2_1  | INFO :
    client2_1  | INFO :      Received: get_parameters message 84fc4734-03f0-4ea0-beba-a4d1a0be0d39
    client2_1  | INFO :      Sent reply
    server_1   | WARNING :   Failed to receive initial parameters from the client. Empty initial parameters will be used.
    server_1   | INFO :      Evaluating initial global parameters
    server_1   | INFO :
    server_1   | INFO :      [ROUND 1]
    server_1   | INFO :      configure_fit: strategy sampled 2 clients (out of 2)
    client2_1  | INFO :
    client2_1  | INFO :      Received: train message cf970480-48be-4273-8769-4e00e3c7e2f3
    client1_1  | INFO :
    client1_1  | INFO :      Received: train message 11ff4979-be06-4bd5-9564-2871a6f5d566
    client1_1  | INFO :      Sent reply
    client2_1  | INFO :      Sent reply
    server_1   | INFO :      aggregate_fit: received 2 results and 0 failures
    server_1   | INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
    client2_1  | INFO :
    client2_1  | INFO :      Received: evaluate message 8c10d9b6-8d81-4595-9301-ba794ce549bf
    client1_1  | INFO :
    client1_1  | INFO :      Received: evaluate message fa64b648-2b74-4412-89be-d057646f7acf
    client1_1  | INFO :      Sent reply
    client2_1  | INFO :      Sent reply
    server_1   | INFO :      aggregate_evaluate: received 2 results and 0 failures
    server_1   | INFO :
    ```

### 4. MTL Model Improvement: Ensemble Stacking using Support Vector Machines (SVM)

- **Running Ensemble Model:**
  - Navigate to the `main_mimic/ensemble_model_demo` directory.
  - Run the Docker orchestrator script using Docker Compose:
    ```bash
    docker-compose up
    ```
  - Expected output:
    ```
    svm_1  | Accuracy of the first level MTL model for Diabetes: 67.61%
    svm_1  | Accuracy of the stacked SVM model for Diabetes: 74.31%
    svm_1  | Accuracy of the first level MTL model for CKD: 79.04%
    svm_1  | Accuracy of the stacked SVM model for CKD: 81.19%

    ```

