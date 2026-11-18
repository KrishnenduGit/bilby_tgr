from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.conversion import _generate_all_cbc_parameters

def convert_to_lal_qbh_parameters(parameters):

    converted_parameters, added_keys = convert_to_lal_binary_black_hole_parameters(
        parameters
    )

    if not any(
        [
            key in converted_parameters
            for key in ["qbh_alpha"]
        ]
    ):
        converted_parameters["qbh_alpha"] = 0
        added_keys = added_keys + ["qbh_alpha"]
        return converted_parameters, added_keys

    return converted_parameters, added_keys


def generate_all_lal_qbh_parameters(sample, likelihood=None, priors=None, npool=1):
    
    
    waveform_defaults = {
        "reference_frequency": 50.0,
        "waveform_approximant": "IMRPhenomPv2",
        "minimum_frequency": 20.0,
    }
    output_sample = _generate_all_cbc_parameters(
        sample,
        defaults=waveform_defaults,
        base_conversion=convert_to_lal_qbh_parameters,
        likelihood=likelihood,
        priors=priors,
        npool=npool,
    )
    return output_sample

