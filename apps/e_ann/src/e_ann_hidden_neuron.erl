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
-export([start_link/0, add_input/2, activate_neuron/1, calculate_output/2]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {weight=0.0, input_list=[], output=0.0, activation=0.0}).

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

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    Weight = e_ann_math:generate_random_weight(),
    log4erl:log(info, "Starting (~p) Hidden neuron with weight of ~p~n",
                [self(), Weight]),
    State = #state{weight=Weight},
    {ok, State}.

handle_call({calculate_output, TargetPids}, _From, State) ->
    Input = State#state.activation,
    Weight = State#state.weight,
    io:format("Input~p weight~p~n",[Input,Weight]),
    Output = Input * Weight,
    NewState = State#state{output=Output},
    [ e_ann_output_neuron:add_input(Pid, Output) || Pid <- TargetPids ],
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
