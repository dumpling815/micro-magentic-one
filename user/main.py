import os, time, httpx
from common.request_schema import InvokeResult, deserialize_messages
from autogen_agentchat.base import TaskResult

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT"))
RETRIES = int(os.getenv("RETRIES"))             # 실패 시 추가 재시도 횟수


def agent_health_check():
    # Check if all agents are healthy
    # TODO: health_check는 orchestrator에 둬야하는게 맞지 않나. -> 점검 필요.
    HEALTH_PATH = os.getenv("HEALTH_PATH")
    agents = ["filesurfer", "websurfer", "coder", "orchestrator"]
    agent_urls = [os.getenv(agent.upper() + "_URL") + HEALTH_PATH for agent in agents]
    try:
        with httpx.Client(timeout=5.0) as client:
            for agent_url in agent_urls:
                response = client.get(agent_url)
                print(f"Agent {agent_url}\tresponse : {response}")
                if response.status_code != 200:
                    print(f"Agent {agent_url} health check failed: {response.text}")
                    return False
    except httpx.HTTPError as e:
        print(f"Error during health check: {e}")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:8080/health")
            if response.status_code != 200:
                print(f"Computerterminal health check failed: {response.text}")
                return False
    except httpx.HTTPError as e:
        print(f"Error during health check(computer terminal): {e}")
        return False
    return True
        

def main():
    setup_trial = 0
    while True:
        try:
            if agent_health_check():
                print("All agents are ready.")
                break
            print("One or more agents are not ready. Retrying...")
            setup_trial += 1
            if setup_trial == RETRIES:
                print("Agents are not responding after multiple attempts. Exiting.")
                return
            time.sleep(5)
        except Exception as e:
            print(f"Error during agent health check: {e}")
            setup_trial += 1
            time.sleep(5)
    
    # Start the main loop if all agents are ready
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        e2e_time_per_request_ms = []
        orchestrate_time_per_request_ms = []
        while True:
            try:
                user_input = input("Your request: ")
                if user_input.lower() in ["exit", "quit", "q", "bye", "quit()", "exit()"]:
                    print("Terminating Agent System...")
                    # Clean up code might be implemented here if needed (Docker Container 정리)
                    # 필요시 Docker Container에게 /kill url로 요청
                    break
                elif user_input == "":
                    print("Task cannot be empty. Please enter a valid request.")
                    break

                # Orchestration Request
                start_time_perf = time.perf_counter()
                final_response: httpx.Response = client.post("http://localhost:8080/start", json={"query": user_input}) # Httpx를 통해 요청할 때 Json으로 직렬화 필요.
            
            except httpx.RequestError as e:
                print(f"Request failed: {e}")
            except (KeyboardInterrupt, EOFError):
                print("Terminating Agent System...")
                break

            # Deserialization
            # 이상적으로는 InvokeResult, TaskResult의 처리도 app.py에 두는게 맞지만, 디버깅 편의를 위해 main.py에 위치.
            print(f"final_response status: {final_response}")
            invoke_result = final_response.json()
            invoke_result = InvokeResult(
                status=invoke_result.get("status"),
                response=invoke_result.get("response"),
                elapsed=invoke_result.get("response")
            )

            messages: list[dict] = invoke_result.response.get("messages")
            stop_reason: str = invoke_result.response.get("stop_reason")
            # Latency Measurement
            end_time_perf = time.perf_counter()
            e2e_time_per_request_ms.append(int((end_time_perf - start_time_perf)*1000))
            orchestrate_time_per_request_ms.append(invoke_result.elapsed.get("orchestration_latency_ms", 0))
            # Display
            print("####################################")
            print(f"Final Message from Micro Magentic-One System:\n-> Source:{messages[-1].get('source')} {messages[-1].get('content') if messages else 'No message returned'}")
            print("####################################")
            print(f"E2E Latency(ms): {e2e_time_per_request_ms[-1]}\nOrchestration Latency(ms): {orchestrate_time_per_request_ms[-1]}")
            print("####################################")
            print(f"Enter 'y' if you want to see the full conversation history, anything else to continue.",end=" ")
            if input().lower() == 'y':
                print(f"Stop reason: {stop_reason}")
                if messages:
                    for message in messages:
                        print("####################################")
                        print(f"Type:\t{message.get('type')}")
                        print(f"ID:\t{message.get('id')}")
                        print(f"Source:\t{message.get('source')}")
                        print(f"Content:\t{message.get('content')}")
            
            #task_result = deserialize_task_result(invoke_result=invoke_result)
            #task_result = TaskResult(**(invoke_result.response))
            #task_result: TaskResult = invoke_result.response if isinstance(invoke_result.response, TaskResult) else None 

if __name__ == "__main__":
    main()