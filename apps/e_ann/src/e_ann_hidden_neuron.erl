%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% Hidden neuron module.
%%% @end
%%% Created : 10 Mar 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_hidden_neuron).

-behaviour(gen_server).

%% API
-export([start_link/0, add_input/2, activate_neuron/1,
         calculate_output/2, init_weights/2, get_input_list/1]).

-export([forward_output/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {weights=[], input_list=[], outputs=[], activation=0.0}).

%%%===================================================================
%%% API
%%%===================================================================
start_link() ->
    gen_server:start_link(?MODULE, [], []).

add_input(NeuronPid, Input) ->
    gen_server:call(NeuronPid, {add_to_input_list, Input}).

activate_neuron(NeuronPid) ->
    gen_server:call(NeuronPid, activate_neuron).

calculate_output(NeuronPid, TargetPids) ->
    gen_server:call(NeuronPid, {calculate_output, TargetPids}).

init_weights(NeuronPid, Count) ->
    gen_server:call(NeuronPid, {init_weights, Count}).

get_input_list(NeuronPid) ->
    gen_server:call(NeuronPid, get_input_list).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    log4erl:log(info, "Starting (~p) Hidden neuron~n",
                [self()]),
    State = #state{weights=[]},
    {ok, State}.

handle_call(get_input_list, _From, State) ->
    {reply, {ok, State#state.input_list}, State};
handle_call({init_weights, Count}, _From, State) ->
    Weights = e_ann_math:generate_random_weights(Count),
    NewState = State#state{weights=Weights},
    log4erl:log(info, "(~p) initialized with weights ~p~n",[self(), Weights]),
    {reply, ok, NewState};
handle_call({calculate_output, TargetPids}, _From, State) ->
    Input = State#state.activation,
    Weights = State#state.weights,
    Outputs = [ Input * Weight || Weight <- Weights ],
    NewState = State#state{outputs=Outputs},
    forward_output(Outputs, TargetPids),
    {reply, ok, NewState};
handle_call({add_to_input_list, Input}, _From, State) ->
    InputList = State#state.input_list,
    NewInputList = [Input | InputList],
    log4erl:log(info, "(~p) added ~p to input_list~n",[self(), Input]),
    NewState = State#state{input_list=NewInputList},
    {reply, ok, NewState};
handle_call(activate_neuron, _From, State) ->
    Inputs = State#state.input_list,
    Activation = e_ann_math:activation(Inputs),
    log4erl:log(info, "(~p) activated with value of:~p~n",
                [self(), Activation]),
    NewState = State#state{activation=Activation},
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

forward_output([], []) ->
    ok;
forward_output(Outputs, TargetNeurons) ->
    e_ann_output_neuron:add_input(hd(TargetNeurons), hd(Outputs)),
    forward_output(tl(Outputs), tl(TargetNeurons)).
