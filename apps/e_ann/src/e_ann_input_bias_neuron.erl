%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% Input layer bias neuron for output layer with static input of 1.
%%% @end
%%% Created : 02 July 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_input_bias_neuron).

-behaviour(gen_server).

%% API
-export([start_link/0, feed_forward/2, calculate_gradient/2]).

-export([get_weights/1, init_weights/2, set_weights/2, update_weights/3]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).
-define(INPUT, 1).

-record(state, {weights=[],
                feedforward_values=[],
                weight_deltas=[],
                gradient=0.0}).

%%%===================================================================
%%% API
%%%===================================================================
start_link() ->
    gen_server:start_link(?MODULE, [], []).

feed_forward(NeuronPid, TargetPids) ->
    gen_server:call(NeuronPid, {feed_forward, TargetPids}).

calculate_gradient(NeuronPid, Delta) ->
    gen_server:call(NeuronPid, {calculate_gradient, Delta}).

init_weights(NeuronPid, Count) ->
    gen_server:call(NeuronPid, {init_weights, Count}).

update_weights(NeuronPid, LearningRate, Momentum) ->
    gen_server:call(NeuronPid, {update_weights, LearningRate, Momentum}).

get_weights(NeuronPid) ->
    gen_server:call(NeuronPid, get_weights).

set_weights(NeuronPid, Weights) ->
    gen_server:call(NeuronPid, {set_weights, Weights}).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    log4erl:info("Starting input bias neuron with pid:(~p)~n", [self()]),
    State = #state{weights=[]},
    {ok, State}.

handle_call({feed_forward, TargetPids}, _From, State) ->
    Weights = State#state.weights,
    FeedForwardValues = [ ?INPUT * Weight || Weight <- Weights ],
    NewState = State#state{feedforward_values=FeedForwardValues},
    e_ann_input_neuron:forward_output(FeedForwardValues, TargetPids),
    {reply, ok, NewState};
handle_call({calculate_gradient, Delta}, _From, State) ->
    Gradient = ?INPUT * Delta,
    log4erl:info("Input bias neuron (~p) gradient:~p~n", [self(), Gradient]),
    NewState = State#state{gradient=Gradient},
    {reply, ok, NewState};
handle_call({update_weights, LearningRate, Momentum}, _From, State) ->
    Gradient = State#state.gradient,
    WeightDeltas = State#state.weight_deltas,
    Weights = State#state.weights,
    NewWeightDeltas = [ (Gradient * LearningRate) + (Weight * Momentum) ||
                       Weight <- WeightDeltas ],
    UpdatedWeights = e_ann_math:update_weights(Weights, NewWeightDeltas),
    log4erl:info("Input bias neuron (~p) updated weights:~p~n",
                 [self(), UpdatedWeights]),
    NewState = State#state{weight_deltas=NewWeightDeltas},
    FinalState = NewState#state{weights=UpdatedWeights},
    {reply, ok, FinalState};
handle_call(get_weights, _From , State) ->
    Weights = State#state.weights,
    {reply, Weights, State};
handle_call({init_weights, Count}, _From, State) ->
    Weights = e_ann_math:generate_random_weights(Count),
    WeightDeltas = e_ann_math:init_weight_deltas(Count),
    NewState = State#state{weights=Weights},
    FinalState = NewState#state{weight_deltas=WeightDeltas},
    log4erl:info("Input bias neuron (~p) initialized weights ~p~n",
                 [self(), Weights]),
    {reply, ok, FinalState};
handle_call({set_weights, Weights}, _From, State) ->
    NewState = State#state{weights=Weights},
    log4erl:info("Input bias neuron (~p) weights set to:~p~n",
                 [self(), Weights]),
    {reply, ok, NewState};
handle_call(_Request, _From, State) ->
    Reply = ok,
    {reply, Reply, State}.

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

