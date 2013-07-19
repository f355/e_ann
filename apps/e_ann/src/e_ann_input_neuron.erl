%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module spawns input neurons that receive an input upon init.
%%% Random weights are then assigned and feedforward commences.
%%% @end
%%% Created :  9 Mar 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_input_neuron).

-behaviour(gen_server).

%% API
-export([start_link/1, feed_forward/2, init_weights/2,
         forward_output/2, calculate_gradient/2, update_weights/3]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {input=0.0,
                weights=[],
                gradient=0.0,
                weight_deltas=[],
                feedforward_values=[]}).

%%%===================================================================
%%% API
%%%===================================================================
start_link(Args) ->
    gen_server:start_link(?MODULE, [Args], []).

feed_forward(NeuronPid, TargetPids) ->
    gen_server:call(NeuronPid, {feed_forward, TargetPids}).

init_weights(NeuronPid, Count) ->
    gen_server:call(NeuronPid, {init_weights, Count}).

calculate_gradient(NeuronPid, Delta) ->
    gen_server:call(NeuronPid, {calculate_gradient, Delta}).

update_weights(NeuronPid, LearningRate, Momentum) ->
    gen_server:call(NeuronPid, {update_weights, LearningRate, Momentum}).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([Input]) ->
     log4erl:info("Starting input neuron with pid:(~p) and input:~p ~n",
                 [self(), Input]),
    State = #state{input=Input},
    {ok, State}.

handle_call({init_weights, Count}, _From, State) ->
    Weights = e_ann_math:generate_random_weights(Count),
    WeightDeltas = e_ann_math:init_weight_deltas(Count),
    NewState = State#state{weights=Weights},
    FinalState = NewState#state{weight_deltas=WeightDeltas},
    log4erl:info("Input neuron (~p) initialized weights~p~n",
                 [self(), Weights]),
    {reply, ok, FinalState};
handle_call({feed_forward, TargetPids}, _From, State) ->
    Input = State#state.input,
    Weights = State#state.weights,
    FeedForwardValues = [ Input * Weight || Weight <- Weights ],
    NewState = State#state{feedforward_values=FeedForwardValues},
    forward_output(FeedForwardValues, TargetPids),
    {reply, ok, NewState};
handle_call({calculate_gradient, Delta}, _From, State) ->
    Input = State#state.input,
    Gradient = Input * Delta,
    log4erl:info("Input neuron (~p) gradient:~p~n", [self(), Gradient]),
    NewState = State#state{gradient=Gradient},
    {reply, ok, NewState};
handle_call({update_weights, LearningRate, Momentum}, _From, State) ->
    Gradient = State#state.gradient,
    WeightDeltas = State#state.weight_deltas,
    Weights = State#state.weights,
    NewWeightDeltas = [ (Gradient * LearningRate) + (Weight * Momentum) ||
                       Weight <- WeightDeltas ],
    UpdatedWeights = e_ann_math:update_weights(Weights, NewWeightDeltas),
    log4erl:info("Input neuron (~p) updated weights:~p~n",
                 [self(), UpdatedWeights]),
    NewState = State#state{weight_deltas=NewWeightDeltas},
    FinalState = NewState#state{weights=UpdatedWeights},
    {reply, ok, FinalState};
handle_call(_Request, _From, State) ->
    {reply, ok, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%%===================================================================
%%% Internal functions
%%%===================================================================

forward_output([], []) ->
    ok;
forward_output(Values, TargetNeurons) ->
    e_ann_hidden_neuron:add_input(hd(TargetNeurons), hd(Values)),
    forward_output(tl(Values), tl(TargetNeurons)).
