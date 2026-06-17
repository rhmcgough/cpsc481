# Ryan McGough
from probability4e import *

T, F = True, False

class Diagnostics:
    """ Use a Bayesian network to diagnose between three lung diseases """

    def __init__(self):
        # Initializing the Bayesian Network
        self.bn = BayesNet([ 

            # P(Asia=T) = 0.01
            ('Asia',       '',                  0.01),

            # P(Smoking=T) = 0.5
            ('Smoking',    '',                  0.5),

            # TB: P(TB=T | Asia)
            ('TB',         'Asia',              {T: 0.05, F: 0.01}),

            # P(Cancer=T | Smoking)
            ('Cancer',     'Smoking',           {T: 0.1, F: 0.01}),

            # P(Bronchitis=T | Smoking)
            ('Bronchitis', 'Smoking',           {T: 0.6, F: 0.3}),


            # TBorC: P(TBorC=T | TB, Cancer)  — deterministic OR
            ('TBorC',      'TB Cancer',         {(T, T): 1.0,
                                                 (T, F): 1.0,
                                                 (F, T): 1.0,
                                                 (F, F): 0.0}),


            # P(Xray=T | TBorC)
            ('Xray',       'TBorC',             {T: 0.99, F: 0.05}),


            # P(Dyspnea=T | TBorC, Bronchitis)
            ('Dyspnea',    'TBorC Bronchitis',  {(T, T): 0.9,
                                                 (T, F): 0.7,
                                                 (F, T): 0.8,
                                                 (F, F): 0.1}),
        ])

    def diagnose (self, asia, smoking, xray, dyspnea):
        
        # Initialize dictionary
        cases = {}

        if asia != "NA":
            cases['Asia'] = (asia == 'Yes')

        if smoking != "NA":
            cases['Smoking'] = (smoking == 'Yes')

        if xray != "NA":
            cases['Xray'] = (xray == 'Abnormal')

        if dyspnea != "NA":
            cases['Dyspnea'] = (dyspnea == 'Present')


        # Query each disease
        p_tb         = enumeration_ask('TB',         cases, self.bn)[T]
        p_cancer     = enumeration_ask('Cancer',     cases, self.bn)[T]
        p_bronchitis = enumeration_ask('Bronchitis', cases, self.bn)[T]


        # Pick the most likely disease
        candidates = [
            ("TB",         p_tb),
            ("Cancer",     p_cancer),
            ("Bronchitis", p_bronchitis),
        ]
        disease, probability = max(candidates, key=lambda x: x[1])

 
        return [disease, probability]