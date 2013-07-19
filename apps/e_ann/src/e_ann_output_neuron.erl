%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module spawns output neurons. After a output neuron has calculated
%%% it's own node delta it initiates backpropagation.
%%% @end
%%% Created : 16 Mar 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_output_neuron).

-behaviour(gen_server).

%% API
-export([start_link/1, add_input/2, activate_neuron/1,
        sum/1, calculate_error/1, calculate_node_delta/1]).

-export([get_node_delta/1, backpropagate_with_bias/3]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-record(state, {global_error=[],
                error=0.0,
                ideal_output=0.0,
                inputs=[],
                sum=0.0,
                output=0.0,
                node_delta=0.0}).

%%%===================================================================
%%% API
%%%===================================================================
start_link(Args) ->
    gen_server:start_link(?MODULE, [Args], []).

add_input(NeuronPid, Input) ->
    gen_server:call(NeuronPid, {add_to_inputs, Input}).

activate_neuron(NeuronPid) ->
    gen_server:call(NeuronPid, activate_neuron).

sum(NeuronPid) ->
    gen_server:call(NeuronPid, sum).

calculate_error(NeuronPid) ->
    gen_server:call(NeuronPid, calculate_error).

calculate_node_delta(NeuronPid) ->
    gen_server:call(NeuronPid, calculate_node_delta).

get_node_delta(NeuronPid) ->
    gen_server:call(NeuronPid, get_node_delta).

backpropagate_with_bias(NeuronPid, Layer, HBias) ->
    gen_server:call(NeuronPid, {backpropagate, Layer, HBias}).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([Ideal]) ->
    log4erl:info("Starting output neuron with pid:(~p) - ideal output:~p~n",
                [self(), Ideal]),
    State = #state{ideal_output=Ideal},
    {ok, State}.
handle_call(calculate_node_delta, _From, State) ->
    Output = State#state.output,
    Error = State#state.error,
    NodeDelta = e_ann_math:output_node_delta(Error, Output),
    log4erl:info("Output neuron (~p) has a node delta of:~p~n",
                 [self(), NodeDelta]),
    NewState = State#state{node_delta=NodeDelta},
    {reply, ok, NewState};
handle_call(calculate_error, _From, State) ->
    Output = State#state.output,
    Ideal = State#state.ideal_output,
    GlobalErrs = State#state.global_error,
    Error = e_ann_math:linear_error(Output, Ideal),
    NewGlobalErrs = [Error | GlobalErrs],
    GlobalError = e_ann_math:mse(NewGlobalErrs),
    log4erl:info("Output neuron (~p) has a global error of:~p~n",
                 [self(), GlobalError]),
    NewState = State#state{global_error=GlobalError},
    FinalState = NewState#state{error=Error},
    {reply, ok, FinalState};
handle_call(sum, _From, State) ->
    Inputs = State#state.inputs,
    Sum = lists:sum(Inputs),
    log4erl:info("Output neuron (~p) sum:~p~n", [self(), Sum]),
    NewState = State#state{sum=Sum},
    {reply, ok, NewState};
handle_call(activate_neuron, _From, State) ->
    Sum = State#state.sum,
    Output = e_ann_math:sigmoid(Sum),
    log4erl:info("Output neuron (~p) output:~p~n", [self(), Output]),
    NewState = State#state{output=Output},
    {reply, ok, NewState};
handle_call({add_to_inputs, Input}, _From, State) ->
    Inputs = State#state.inputs,
    NewInputs = [Input | Inputs],
    log4erl:info("Output neuron (~p) added ~p to inputs~n",[self(), Input]),
    NewState = State#state{inputs=NewInputs},
    {reply, ok, NewState};
handle_call(get_node_delta, _From, State) ->
    {reply, State#state.node_delta, State};
handle_call({backpropagate, Layer, HBias}, _From, State) ->
    Delta = State#state.node_delta,
    [ e_ann_hidden_neuron:calculate_node_delta(Pid ,Delta) || Pid <- Layer ],
    [ e_ann_hidden_neuron:calculate_gradient(Pid, Delta) || Pid <- Layer ],
    e_ann_hidden_bias_neuron:calculate_gradient(HBias, Delta),
    {reply, ok, State};
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
