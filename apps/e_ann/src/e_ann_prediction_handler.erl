%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module starts a production network with trained weights that
%%% can be used for e.g. predictions and pattern recognition.
%%% @end
%%% Created :  23 July 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_prediction_handler).

-export([init_xor_network/1, set_xor_weights/2]).

init_xor_network(Architecture) ->
    [ICount, HCount, OCount] = Architecture,
    [{_,IBSup},{_, HBSup},{_,HSup},
     {_,OSup},{_,ISup}] = e_ann_network:get_sup_pids(),
    ISupFun = fun e_ann_input_neuron_sup:add_child/1,
    HSupFun = fun e_ann_hidden_neuron_sup:add_child/1,
    OSupFun = fun e_ann_output_neuron_sup:add_child/1,
    SetWeightIFun = fun e_ann_input_neuron:set_weights/2,
    IL = e_ann_network:spawn_neurons(ICount, ISup, ISupFun, []),
    HL = e_ann_network:spawn_neurons(HCount, HSup, HSupFun, []),
    OL = e_ann_network:spawn_neurons(OCount, OSup, OSupFun, []),
    {ok, IBias} = e_ann_input_bias_neuron_sup:add_child(IBSup),
    {ok, HBias} = e_ann_hidden_bias_neuron_sup:add_child(HBSup),
    [IL, HL, OL, IBias, HBias].

set_xor_weights(Network, Weights) ->
    [{_,IW1}, {_, IW2}, {_,HW1}, {_,HW2}, {_,IBW}, {_,HBW}] = Weights,
    [IL, HL, OL, IBias, HBias] = Network,
    [I1, I2] = IL,
    [H1, H2] = HL,
    [O1] = OL,
    e_ann_input_neuron:set_weights(I1, IW1),
    e_ann_input_neuron:set_weights(I2, IW2),
    e_ann_hidden_neuron:set_weights(H1, HW1),
    e_ann_hidden_neuron:set_weights(H2, HW2),
    e_ann_input_bias_neuron:set_weights(IBias, IBW),
    e_ann_hidden_bias_neuron:set_weights(HBias, HBW),
    ok.
