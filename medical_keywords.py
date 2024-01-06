medical_keywords = [
    "symptom", "illness", "disease", "treatment", "medication", "health", "diagnosis",
    "cold", "fever", "headache", "cough", "pain", "infection", "allergy", "nausea", 
    "vomiting", "diarrhea", "fatigue", "injury", "wound", "rash", "inflammation",
    "swelling", "bruise", "bleeding", "burn", "fracture", "sprain", "strain",
    "depression", "anxiety", "insomnia", "diabetes", "hypertension", "asthma",
    "arthritis", "cancer", "heart disease", "stroke", "flu", "pneumonia",
    "bronchitis", "acne", "eczema", "psoriasis", "dermatitis", "allergies",
    "obesity", "anemia", "cholesterol", "thyroid", "migraine", "dizziness",
    "menstrual", "pregnancy", "contraception", "influenza", "tuberculosis",
    "malaria", "HIV", "AIDS", "hepatitis", "cirrhosis", "kidney stones",
    "gastroenteritis", "ulcer", "constipation", "indigestion", "dehydration",
    "osteoporosis", "sclerosis", "autism", "epilepsy", "dementia",
    "schizophrenia", "bipolar disorder", "oral health", "dental", "vision",
    "hearing", "ENT", "pediatric", "geriatric", "nutrition", "exercise",
    "vaccine", "immunization", "first aid", "emergency", "surgery",
    "rehabilitation", "therapy", "mental health", "wellness", "fitness",
    "lifestyle", "addiction", "recovery", "preventive care"
    "immunodeficiency", "leukemia", "lymphoma", "sarcoma", "melanoma",
    "anorexia", "bulimia", "malnutrition", "vitamin deficiency", "iron deficiency",
    "hypoglycemia", "hyperglycemia", "endocrinology", "polycystic ovary syndrome",
    "endometriosis", "fibroids", "prostatitis", "erectile dysfunction", "infertility",
    "urology", "nephrology", "glomerulonephritis", "urinary tract infection",
    "cystitis", "bladder stones", "renal failure", "dialysis", "liver disease",
    "gallstones", "cholecystitis", "pancreatitis", "gastroesophageal reflux disease",
    "irritable bowel syndrome", "Crohn's disease", "ulcerative colitis",
    "diverticulitis", "hemorrhoids", "anal fissure", "cardiology", "arrhythmia",
    "coronary artery disease", "myocardial infarction", "congestive heart failure",
    "hypotension", "peripheral artery disease", "venous thrombosis", 
    "pulmonary embolism", "chronic obstructive pulmonary disease", "asthma",
    "pulmonary hypertension", "pneumothorax", "pleurisy", "otitis media",
    "tinnitus", "vertigo", "meningitis", "encephalitis", "multiple sclerosis",
    "Parkinson's disease", "Alzheimer's disease", "peripheral neuropathy",
    "sciatica", "carpal tunnel syndrome", "muscular dystrophy", "myasthenia gravis",
    "osteomyelitis", "septic arthritis", "gout", "lupus", "rheumatoid arthritis",
    "psoriatic arthritis", "dermatology", "hives", "impetigo", "cellulitis",
    "herpes zoster", "herpes simplex", "warts", "fungal infections", "ringworm",
    "scabies", "lice", "bedsores", "hair loss", "alopecia", "nail fungus",
    "oral ulcers", "gingivitis", "periodontitis", "tooth decay", "toothache",
    "ophthalmology", "conjunctivitis", "cataracts", "glaucoma", "macular degeneration",
    "retinopathy", "strabismus", "ear infection", "sinusitis", "pharyngitis",
    "laryngitis", "hoarseness", "sleep apnea", "narcolepsy", "restless leg syndrome",
    "sleep disorders", "sexual health", "sexually transmitted infections",
    "chlamydia", "gonorrhea", "syphilis", "genital herpes", "human papillomavirus",
    "trichomoniasis", "pubic lice", "scrotal swelling", "breast cancer",
    "mastitis", "menopause", "hormone replacement therapy", "osteopenia",
    "kyphosis", "scoliosis", "spinal stenosis", "herniated disc", "spondylolisthesis",
    "amputation", "burn care", "wound care", "plastic surgery", "reconstructive surgery",
    "transplant surgery", "anesthesiology", "radiology", "ultrasound", 
    "MRI", "CT scan", "X-ray", "nuclear medicine", "pathology", "biopsy",
    "blood test", "urine test", "gene therapy", "stem cell therapy", "clinical trials",
    "public health", "epidemiology", "vaccination", "global health", "health policy",
    "telemedicine", "eHealth", "digital health", "health informatics", "patient safety"
    "congenital heart defect", "cardiomyopathy", "angina", "pericarditis", "endocarditis",
    "rheumatic fever", "varicose veins", "phlebitis", "Raynaud's disease", "peripheral edema",
    "emphysema", "bronchiectasis", "pleural effusion", "asbestosis", "sarcoidosis",
    "cystic fibrosis", "silicosis", "pulmonary fibrosis", "sleep disorders", "snoring",
    "tonsillitis", "deviated septum", "rhinitis", "polyps", "sinus infection",
    "laryngeal cancer", "voice disorders", "stuttering", "aphasia", "dysarthria",
    "brain tumor", "concussion", "cerebral palsy", "hydrocephalus", "spina bifida",
    "bell's palsy", "Guillain-Barre syndrome", "neuralgia", "dysphagia", "vertigo",
    "otitis externa", "Meniere's disease", "presbycusis", "macular hole", "retinal detachment",
    "uveitis", "keratitis", "conjunctivitis", "stye", "glaucoma", "cataract",
    "color blindness", "hyperopia", "myopia", "astigmatism", "presbyopia",
    "tooth erosion", "halitosis", "oral candidiasis", "stomatitis", "oral herpes",
    "periodontal disease", "dental abscess", "bruxism", "TMJ disorders", "malocclusion",
    "esophagitis", "Barrett's esophagus", "hiatal hernia", "achalasia", "gallbladder disease",
    "peptic ulcer", "celiac disease", "intestinal ischemia", "colitis", "intestinal polyps",
    "anal fistula", "proctitis", "rectal prolapse", "hematochezia", "portal hypertension",
    "hepatic encephalopathy", "ascites", "cholangitis", "liver abscess", "pancreatic cancer",
    "cystitis", "urethritis", "bladder cancer", "kidney infection", "glomerulonephritis",
    "polycystic kidney disease", "renal cell carcinoma", "kidney cysts", "urosepsis",
    "prostate cancer", "benign prostatic hyperplasia", "testicular torsion", "hydrocele",
    "varicocele", "penile cancer", "phimosis", "priapism", "erectile dysfunction",
    "endometriosis", "uterine fibroids", "cervical dysplasia", "ovarian cysts", "ovarian cancer",
    "vaginitis", "vulvodynia", "pelvic inflammatory disease", "menorrhagia", "amenorrhea",
    "dysmenorrhea", "premenstrual syndrome", "gestational diabetes", "preeclampsia",
    "ectopic pregnancy", "postpartum depression", "menopausal symptoms", "hormonal imbalance",
    "osteomalacia", "rickets", "bone cancer", "bone spurs", "bunions", "corns and calluses",
    "hammertoe", "heel pain", "ingrown toenails", "neuroma", "plantar fasciitis",
    "achilles tendinitis", "bursitis", "frozen shoulder", "rotator cuff injury",
    "tendinitis", "muscle cramps", "muscle strain", "muscular dystrophy", "spasticity",
    "fibromyalgia", "systemic lupus erythematosus", "scleroderma", "polymyositis",
    "dermamyositis", "psoriatic arthritis", "ankylosing spondylitis", "vasculitis",
    "lichen planus", "seborrheic dermatitis", "vitiligo", "pemphigus", "bullous pemphigoid",
    "rosacea", "acne rosacea", "moles", "skin tags", "actinic keratosis", "sebaceous cysts",
    "lipoma", "keratosis pilaris", "atopic dermatitis", "contact dermatitis",
    "drug rash", "heat rash", "intertrigo","stomachache","stomach"
    
]