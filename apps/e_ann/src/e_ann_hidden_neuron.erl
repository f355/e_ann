%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% Hidden neuron module.
%%% @end
%%% Created : 10 June 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_hidden_neuron).

-behaviour(gen_server).

%% API
-export([start_link/0, add_input/2, activate_neuron/1,
         feed_forward/2, init_weights/2, sum/1]).

-export([forward_output/2, calculate_node_delta/2, update_weights/3,
         calculate_gradient/2, backpropagate_with_bias/3]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {weights=[],
                weight_deltas=[],
                inputs=[],
                sum=[],
                feedforward_values=[],
                output=0.0,
                gradient=0.0,
                node_delta=0.0}).

%%%===================================================================
%%% API
%%%===================================================================
start_link() ->
    gen_server:start_link(?MODULE, [], []).

add_input(NeuronPid, Input) ->
    gen_server:call(NeuronPid, {add_to_inputs, Input}).

activate_neuron(NeuronPid) ->
    gen_server:call(NeuronPid, activate_neuron).

feed_forward(NeuronPid, TargetPids) ->
    gen_server:call(NeuronPid, {feed_forward, TargetPids}).

init_weights(NeuronPid, Count) ->
    gen_server:call(NeuronPid, {init_weights, Count}).

sum(NeuronPid) ->
    gen_server:call(NeuronPid, sum).

calculate_node_delta(NeuronPid, Delta) ->
    gen_server:call(NeuronPid, {calculate_node_delta, Delta}).

calculate_gradient(NeuronPid, Delta) ->
    gen_server:call(NeuronPid, {calculate_gradient, Delta}).

backpropagate_with_bias(NeuronPid, Layer, IBias) ->
    gen_server:call(NeuronPid, {backpropagate, Layer, IBias}).

update_weights(NeuronPid, LearningRate, Momentum) ->
    gen_server:call(NeuronPid, {update_weights, LearningRate, Momentum}).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    log4erl:info("Starting hidden neuron with pid:~p~n", [self()]),
    State = #state{weights=[]},
    {ok, State}.

handle_call({calculate_node_delta, Delta}, _From, State) ->
    Sum = State#state.sum,
    Weight = hd(State#state.weights),
    NodeDelta = e_ann_math:interior_node_delta(Sum, Delta, Weight),
    log4erl:info("Hidden neuron (~p) node delta:~p~n", [self(), NodeDelta]),
    NewState = State#state{node_delta=NodeDelta},
    {reply, ok, NewState};
handle_call({calculate_gradient, Delta}, _From, State) ->
    Output = State#state.output,
    Gradient = Output * Delta,
    log4erl:info("Hidden neuron (~p) gradient:~p~n", [self(), Gradient]),
    NewState = State#state{gradient=Gradient},
    {reply, ok, NewState};
handle_call(sum, _From, State) ->
    Inputs = State#state.inputs,
    Sum = lists:sum(Inputs),
    log4erl:info("Hidden neuron (~p) sum:~p~n", [self(), Sum]),
    NewState = State#state{sum=Sum},
    {reply, {ok, Sum}, NewState};
handle_call({init_weights, Count}, _From, State) ->
    Weights = e_ann_math:generate_random_weights(Count),
    WeightDeltas = e_ann_math:init_weight_deltas(Count),
    NewState = State#state{weights=Weights},
    FinalState = NewState#state{weight_deltas=WeightDeltas},
    log4erl:info("Hidden Neuron (~p) initialized weights ~p~n",
                 [self(), Weights]),
    {reply, ok, FinalState};
handle_call({feed_forward, TargetPids}, _From, State) ->
    Output = State#state.output,
    Weights = State#state.weights,
    FeedForwardValues = [ Output * Weight || Weight <- Weights ],
    NewState = State#state{feedforward_values=FeedForwardValues},
    forward_output(FeedForwardValues, TargetPids),
    {reply, ok, NewState};
handle_call({add_to_inputs, Input}, _From, State) ->
    Inputs = State#state.inputs,
    NewInputs = [Input | Inputs],
    log4erl:info("Hidden neuron (~p) added ~p to inputs~n",[self(), Input]),
    NewState = State#state{inputs=NewInputs},
    {reply, ok, NewState};
handle_call(activate_neuron, _From, State) ->
    Sum = State#state.sum,
    Output = e_ann_math:sigmoid(Sum),
    log4erl:info("Hidden neuron (~p) output value:~p~n", [self(), Output]),
    NewState = State#state{output=Output},
    {reply, ok, NewState};
handle_call({backpropagate, Layer, IBias}, _From, State) ->
    Delta = State#state.node_delta,
    [ e_ann_input_neuron:calculate_gradient(Pid, Delta) || Pid <- Layer ],
    e_ann_input_bias_neuron:calculate_gradient(IBias, Delta),
    {reply, ok, State};
handle_call({update_weights, LearningRate, Momentum}, _From, State) ->
    Gradient = State#state.gradient,
    WeightDeltas = State#state.weight_deltas,
    Weights = State#state.weights,
    NewWeightDeltas = [ (Gradient * LearningRate) + (Weight * Momentum) ||
                       Weight <- WeightDeltas ],
    UpdatedWeights = e_ann_math:update_weights(Weights, NewWeightDeltas),
    log4erl:info("Hidden neuron (~p) updated weights:~p~n",
                 [self(), UpdatedWeights]),
    NewState = State#state{weight_deltas=NewWeightDeltas},
    FinalState = NewState#state{weights=UpdatedWeights},
    {reply, ok, FinalState};
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

forward_output([], []) ->
    ok;
forward_output(Values, TargetNeurons) ->
    e_ann_output_neuron:add_input(hd(TargetNeurons), hd(Values)),
    forward_output(tl(Values), tl(TargetNeurons)).
