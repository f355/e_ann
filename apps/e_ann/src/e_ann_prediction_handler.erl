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

-export([init_network/1]).

init_network(Architecture) ->
    [ICount, HCount, OCount, InputBias, HiddenBias] = Architecture,
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


make_prediction(Network, Inputs) ->
    [Il, HL, OL, IBias, HBias] = Network.
