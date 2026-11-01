from types import SimpleNamespace
import flwr as fl
import numpy as np
from fedlegal.federated.adaptive_strategy import CheckpointingFedAdam


def test_fedadam_retains_momentum_when_next_client_delta_is_zero(tmp_path):
    strategy=CheckpointingFedAdam(tmp_path,2,fl.common.ndarrays_to_parameters([np.zeros(1)]),None)
    status=fl.common.Status(code=fl.common.Code.OK,message="")
    def fit(value):
        return [(SimpleNamespace(cid=str(i)),fl.common.FitRes(status,
                fl.common.ndarrays_to_parameters([np.array([value])]),1,{})) for i in range(2)]
    first,_=strategy.aggregate_fit(1,fit(3.0),[])
    initial=fl.common.parameters_to_ndarrays(first)[0][0]
    expected=.01*np.sqrt(1-.99**2)/(1-.9**2)*.3/(np.sqrt(.09)+1e-9)
    assert np.isclose(initial,expected)
    second,_=strategy.aggregate_fit(2,fit(initial),[])
    assert fl.common.parameters_to_ndarrays(second)[0][0] > initial
    assert (tmp_path/"server_round_0002.npz").exists()
