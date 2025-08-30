import os, time, httpx
from RequestSchema import InvokeBody, InvokeResult, Msg
from autogen_agentchat.base import TaskResult

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
ENTRYPOINT_URL = os.getenv("ENTRYPOINT", "http://localhost:8000/start")
RETRIES = int(os.getenv("RETRIES", "1"))             # 실패 시 추가 재시도 횟수

def agent_health_check():
    # Check if all agents are healthy
    agents = ["filesurfer", "websurfer", "coder", "computerterminal", "orchestrator"]  
    try:
        with httpx.Client(timeout=5.0) as client:
            for agent in agents:
                response = client.get(f"http://{agent}:8000/health")
                if response.status_code != 200:
                    print(f"Agent {agent} health check failed: {response.text}")
                    return False
        return True
    except httpx.HTTPError as e:
        print(f"Error during health check: {e}")
        return False

def main():
    setup_trial = 0
    while True:
        try:
            if agent_health_check():
                print("All agents are ready.")
                break
            print("One or more agents are not ready. Retrying...")
            setup_trial += 1
            if setup_trial == 6:
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

                start_time_perf = time.perf_counter()
                user_msg = Msg(type="TextMessage", source="user", content=user_input)
                body = InvokeBody(messages=[user_msg])

                final_response: InvokeResult = client.post(ENTRYPOINT_URL, json=body.model_dump_json()) # Httpx를 통해 요청할 때 Json으로 직렬화 필요.
                result = InvokeResult(**final_response) # 응답 형태는 Json, 역직렬화.
                task_result: TaskResult = result.response if isinstance(result.response, TaskResult) else None 

                # Latency Measurement
                end_time_perf = time.perf_counter()
                e2e_time_per_request_ms.append(int((end_time_perf - start_time_perf)*1000))
                orchestrate_time_per_request_ms.append(result.elapsed.get("orchestration_latency_ms", 0))


                # Display Result of the Entire System
                print("####################################")
                print(f"Final Message from Micro Magentic-One System:\n->  {task_result.messages[-1].content if task_result and task_result.messages else 'No message returned'}")
                print("####################################")
                print(f"E2E Latency(ms): {e2e_time_per_request_ms[-1]}\nOrchestration Latency(ms): {orchestrate_time_per_request_ms[-1]}")
                print("####################################")
                print(f"Enter 'y' if you want to see the full conversation history, anything else to continue.",end="")
                if input().lower() == 'y':
                    if task_result and task_result.messages:
                        print("Full Conversation History:")
                        for msg in task_result.messages:
                            print(f"[{msg.source}]: {msg.content}")
                    else:
                        print("No conversation history available.")
            except httpx.RequestError as e:
                print(f"Request failed: {e}")
            except (KeyboardInterrupt, EOFError):
                print("Terminating Agent System...")
                break

if __name__ == "__main__":
    main()