from demographer import process_tweet
from demographer.indorg_neural import NeuralOrganizationDemographer
from demographer.gender_neural import NeuralGenderDemographer
from demographer.ethnicity_selfreport_neural import EthSelfReportNeuralDemographer

models = [
        NeuralOrganizationDemographer(setup = "full"),
        NeuralGenderDemographer(),
        EthSelfReportNeuralDemographer(balanced=True, model_dir = r'C:\Users\prasr\Documents\Northeastern\RA\Metoo_research\metoo_research\models\ethnicity_selfreport')
    ]