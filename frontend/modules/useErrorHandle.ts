import { useToast } from '@chakra-ui/react';
import { useCallback } from 'react';
import { CombinedError } from 'urql';

const useErrorHandle = () => {
  const toast = useToast();

  return useCallback(
    ({
      error,
      response,
      msg,
    }: {
      error: CombinedError | undefined;
      response: any;
      msg?: {
        title: string;
        description: string;
      };
    }) => {
      if (error) {
        toast({
          title: 'Error Occurred',
          description: error.message.replace('[GraphQL]', ''),
          status: 'error',
          duration: 9000,
          isClosable: true,
        });
        return;
      }

      if (response) {
        toast({
          ...msg,
          status: 'success',
          duration: 9000,
          isClosable: true,
        });
      }
    },
    []
  );
};

export default useErrorHandle;
