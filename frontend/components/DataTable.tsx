import * as React from 'react';
import { Table, Thead, Tbody, Tr, Th, Td, chakra } from '@chakra-ui/react';
import { TriangleDownIcon, TriangleUpIcon } from '@chakra-ui/icons';
import {
  useReactTable,
  flexRender,
  getCoreRowModel,
  ColumnDef,
  SortingState,
  getSortedRowModel,
} from '@tanstack/react-table';

export type DataTableProps<Data extends object> = {
  data: Data[];
  columns: ColumnDef<Data, any>[];
  onClick?(e: any): void;
  defaultIndex?: number;
  selectedIndex?: number;
};

export function DataTable<Data extends object>({
  data,
  columns,
  onClick,
  defaultIndex,
  selectedIndex,
}: DataTableProps<Data>) {
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const table = useReactTable({
    columns,
    data,
    getCoreRowModel: getCoreRowModel(),
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    state: {
      sorting,
    },
  });

  return (
    <Table>
      <Thead>
        {table.getHeaderGroups().map((headerGroup) => (
          <Tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => {
              // see https://tanstack.com/table/v8/docs/api/core/column-def#meta to type this correctly
              const meta: any = header.column.columnDef.meta;
              return (
                <Th
                  key={header.id}
                  onClick={header.column.getToggleSortingHandler()}
                  isNumeric={meta?.isNumeric}
                >
                  {flexRender(
                    header.column.columnDef.header,
                    header.getContext()
                  )}

                  <chakra.span pl="4">
                    {header.column.getIsSorted() ? (
                      header.column.getIsSorted() === 'desc' ? (
                        <TriangleDownIcon
                          aria-label="sorted descending"
                          w={'2.5'}
                          marginTop={'-5px'}
                        />
                      ) : (
                        <TriangleUpIcon
                          aria-label="sorted ascending"
                          w={'2.5'}
                          marginTop={'-5px'}
                        />
                      )
                    ) : (
                      <TriangleDownIcon
                        aria-label="sorted descending"
                        w={'2.5'}
                        marginTop={'-5px'}
                      />
                    )}
                  </chakra.span>
                </Th>
              );
            })}
          </Tr>
        ))}
      </Thead>
      <Tbody>
        {table.getRowModel().rows.map((row) => {
          const isSelected = row?.index === selectedIndex;
          return (
            <Tr
              borderColor={'#fff'}
              borderWidth={2}
              key={row.id}
              style={{
                borderColor: '#fff',
                backgroundColor: isSelected ? '#F6FFF5' : '#fff',
              }}
              onClick={() => {
                onClick?.(row);
              }}
              className="cursor-pointer"
            >
              {row.getVisibleCells().map((cell) => {
                const meta: any = cell.column.columnDef.meta;
                return (
                  <Td
                    key={cell.id}
                    isNumeric={meta?.isNumeric}
                    style={{
                      color: !isSelected ? '#000' : '#00B074',
                    }}
                    className="text-[13px]"
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </Td>
                );
              })}
            </Tr>
          );
        })}
      </Tbody>
    </Table>
  );
}
