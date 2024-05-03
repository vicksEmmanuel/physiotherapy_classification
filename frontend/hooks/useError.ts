import { useToast } from '@chakra-ui/react';

const useError = () => {
  const toast = useToast();
  const showErrorMessage = (title: string, e?: any) => {
    const ReadableError: Array<string> =
      e?.response?.errors?.map((z: any): string => {
        return (
          String(z?.message).charAt(0).toUpperCase() +
            String(z?.message).slice(1) || 'Something Went Wrong'
        );
      }) || [];

    toast({
      title,
      description: ReadableError.join('\n'),
      status: 'error',
      duration: 9000,
      isClosable: true,
    });
  };

  return { showErrorMessage };
};

export default useError;
