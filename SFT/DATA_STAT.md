Task types:
  medical_dialog            4845
  factual                   34
  prevention                20
  medical_safety            18
  nutrition                 12
  symptoms                  12
  health_education          10
  health_advice             10
  body_systems              7

Sources:
  global_medical.txt             4845
  seeds.txt                      123

Summary: 4968
Train: 4720
Validation: 248

Error samples: 162
     7 | seeds.txt | test savoli bilan bir xil
       Example: What is the purpose of the immune system?
       
   155 | global_medical.txt | takror savol
       Example: Q. Can Yasmin birth control pill be used as an emergency contraceptive pill? Hello doctor, My fiancee and I had unprotected sex a few days back, but I did not ejaculate inside her. Just to be on the safer side, we wanted to use the emergency contraceptive pill. But due to some restriction in the country where we live, Plan B or emergency contraceptive pills are not available. I read that Yasmin, which is used as a regular contraceptive pill can be used as an emergency contraceptive pill at a higher dosage. Can Yasmin be used as an emergency contraceptive pill? And at what dosage?