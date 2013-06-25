-module(e_ann_tests).

-include_lib("eunit/include/eunit.hrl").

%% Train a simple XOR.
main_test() ->
    Input = [[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]],
    Ideal = [[0.0], [1.0], [1.0], [0.0]],
    ErrorRate = 0.01,
    InputNeuronNumber = 2,
    HiddenNeuronNumber = 2,
    OutputNeuronNumber = 1,
    Architecture = {InputNeuronNumber, HiddenNeuronNumber, OutputNeuronNumber},
    ?assertEqual(ok, e_ann_handler:train(Input, Ideal, Architecture, ErrorRate)).
