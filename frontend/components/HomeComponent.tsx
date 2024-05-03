import { Spinner } from "@chakra-ui/react";
import { isEmpty } from "lodash";
import { useMemo, useState } from "react";
import DragAndUpload from "./DragAndUpload";

const HomeComponent = () => {
    const [isLoading, setIsLoading] = useState(false)
    const [values, setValues] = useState()

    const decision = useMemo(() => {
        if (isEmpty(values)) {
            return {
                grade: '',
                explanation: ''
            }
        }

        const resultValue = ((values as any)?.result as string)?.replaceAll("\n","").replaceAll("\\","")

        return {
            grade: JSON.parse(resultValue)?.decision??'',
            explanation: JSON.parse(resultValue)?.explanation??''
        }
    },[values])

    return (
        <div className="w-full  h-full flex flex-col md:flex-row items-center justify-center">
           <div className="w-full lg:w-1/2 h-full flex flex-col p-1 md:p-20 md:items-center md:justify-center" >
                <DragAndUpload 
                    isLoading={isLoading} 
                    setIsLoading={setIsLoading} 
                    setValue={(e: any) => {
                        setIsLoading(false)
                        if (e?.result){
                            setValues(e?.result)
                        }
                    }} 
                />
           </div>
           <div className="w-full lg:flex md:w-1/2 h-full flex-col items-center overflow-hidden">
                <div className="w-full h-full bg-white overflow-auto">
                    <div className="mt-[30px] mx-4 font-bold">Your Analysis would show up here</div>
                    {isLoading? (
                        <div className="w-full h-full items-center flex justify-center">
                            <Spinner />
                        </div>
                    ):(
                        <div className="w-full h-full">
                           {isEmpty(decision.grade)? (<></>):(
                             <div className="mt-10 mx-5">
                                <div className="font-bold text-2xl capitalize">Grade: {decision.grade?.replaceAll("Action","").replace("2.","").replace("1.","").replace("3.","")}</div>
                                <div className="font-bold text-lg mt-4">Explanation</div>
                                <div className="text-sm text-slate-600 text-justify">{decision.explanation}</div>
                            </div>
                           )}
                        </div>
                    )}
                </div>
           </div>
        </div>
    )
}

export default HomeComponent;