import { Spinner, useToast } from '@chakra-ui/react';
import { useCallback, useState } from "react";
import { useDropzone } from 'react-dropzone';
import CsvIcon from './CsvIcon';



const DragAndUpload = ({isLoading, setIsLoading, setValue}: {
    isLoading: boolean, 
    setIsLoading: any,
    setValue: any,
}) => {

    const accept = {
        'video/mp4': ['.mp4'],
    };

    const [file, setFile] = useState();
    const toast = useToast();

    const onDrop = useCallback((acceptedFiles: any) => {
        if (!(acceptedFiles?.[0]?.type as string).includes('mp4')) {
        toast({
            title: 'Not A Mp4 file',
            description: 'Please add a mp4 file',
            status: 'error',
            duration: 9000,
            isClosable: true,
        });
        return;
        }

        setFile(acceptedFiles?.[0]);
    }, []);


    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept,
    });
    
    const upload = async () => {
        if (!file) return;

        if (isLoading) return;

        setIsLoading(true);

        const formData = new FormData();
        formData.append('video', file);
        const response = await fetch('http://ec2-3-84-158-161.compute-1.amazonaws.com/predict', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            setValue?.(await response.json())
        } else {
            toast({
                title: "Something went wrong",
                description: "Something didn't go right. Please try again",
                status: 'error'
            })
            setIsLoading(true);
        }
    };


    return (
        <div className="bg-white rounded m-4 p-4">
            <div className="text-md font-extrabold mb-4 mt-5">IMPORT TO ANALYZE YOUR PHYSIOTHERAPY VIDEO</div>
            <div className="text-xs mb-3">
                Import your video and get it analyzed and graded. Note: we use lesser resources therefore 
                it could take a short while to complete processing.
            </div>

            <div className="border-[1px] border-dashed">
                <div className="dropzone-div cursor-pointer" {...getRootProps()}>
                <input className="dropzone-input" {...getInputProps()} />
                <div className="text-center">
                    <div className="">
                    <div className="items-center justify-center  rounded-md flex py-10 mt-5">
                        {!file && (
                        <>
                            {isDragActive ? (
                            <p className="dropzone-content">
                                Release to drop the files here
                            </p>
                            ) : (
                            <div className="items-center justify-center flex flex-col">
                                <CsvIcon />
                                <div className="text-xs">Drag and Drop</div>
                            </div>
                            )}
                        </>
                        )}

                        {file && <CsvIcon color={'#007AFF'} />}
                    </div>
                    </div>
                </div>
                </div>
                <div className="w-full flex items-center justify-center mb-5">
                {file && (
                    <div
                    onClick={() => {
                        if (isLoading) return;
                        upload();
                    }}
                    className="cursor-pointer text-[#007AFF] text-sm bg-[#E9F3FF] text-center py-2 rounded-md w-[150px] self-center"
                    >
                    Import File{' '}
                    {isLoading && <Spinner size={'xs'} color="lightgray" />}
                    </div>
                )}
                </div>
            </div>
        </div>
    )
}

export default DragAndUpload