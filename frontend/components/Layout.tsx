import { useAuthState } from "@/state/useAuthState";
import { Button } from "@chakra-ui/react";
import { signOut, useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { useEffect } from "react";

const SmartSheetLayout  = ({ children }: any) => {

    const { reset,} = useAuthState()
    const {data} = useSession()
    const {push, asPath} = useRouter()

    const {setLink} = useAuthState();


    useEffect(() => {
        if (asPath) {
        setLink(asPath);
        }
    }, [asPath]);

    return (
        <div className="w-full max:h-screen h-screen flex flex-col bg-[#6666e7]">

            <div className="h-[10%] w-full flex flex-row items-center justify-center ">

                <div className="h-full md:h-[20%] w-[40%] md:w-[30%] ">
                    <div className="w-full h-full flex items-center justify-center">
                        <img src="/tick-small.png" alt="logo" className="md:w-[45px] md:h-[45px] w-[20px] h-[20px] nd:mb-3 mb-0" />
                        <div className="text-white text-sm md:text-2xl font-bold md:-m-2 m-0">Physio-Tees</div>
                    </div>
                </div>

                <div className="md:w-full md:h-[60%] h-full w-[40%] ">
                </div>

                {data && (
                    <div className=" md:h-[20%] h-full w-[40%] flex flex-row justify-center items-center">

                        <Button
                            colorScheme="red"
                            className=" text-white flex  md:flex-row flex-col  items-center justify-center w-full hover:bg-[rgba(255,255,255,0.1)] rounded-none"
                            onClick={async () => {
                                await signOut()
                                 reset()
                                push('/')

                            }}
                        >
                            <img src="/icons8_logout_20px.png" alt="logout" className="md:w-[20px] md:h-[20px] w-[15px] h-[15px] mr-3 md:-ml-10 ml-0" />
                            <div className="text-white md:text-lg text-sm">Logout</div>
                        </Button>


                        <Button
                            colorScheme="red"
                            className=" text-white flex md:flex-row flex-col items-center justify-center w-full hover:bg-[rgba(255,255,255,0.1)] rounded-none"
                            onClick={() => {
                                window.location.href = "mailto:vicksemmanuel58@gmail.com";
                            }}
                        >
                            <img src="/icons8_help_20px.png" alt="help" className="md:w-[20px] md:h-[20px] w-[15px] h-[15px] mr-3  md:-ml-10 ml-0" />
                            <div className="text-white  md:text-lg text-sm">Help <span className="invisible">__</span></div>
                        </Button>
                    </div>
                )}

            </div>
                

            <div className="w-full md:h-full h-[85%] bg-gray-200">
                {children}
            </div>
        </div>
    )
}

export default SmartSheetLayout;